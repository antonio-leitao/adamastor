use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Re-export macros from adamastor-macros
pub use adamastor_macros::{prompt, schema, tool};

// ============ Error Handling ============

#[derive(Debug, thiserror::Error)]
pub enum AdamastorError {
    #[error("API error: {0}")]
    Api(String),

    #[error("Tool '{tool}' execution failed: {error}")]
    ToolExecution { tool: String, error: String },

    #[error("Tool '{0}' not found")]
    ToolNotFound(String),

    #[error("Invalid response format: {0}")]
    ParseError(String),

    #[error("Failed to render prompt: {0}")]
    PromptRenderError(String),

    #[error("Rate limit exceeded - waiting would exceed timeout")]
    RateLimit,

    #[error("Maximum function calls ({0}) exceeded")]
    MaxFunctionCalls(u32),

    #[error("File operation failed: {0}")]
    FileOperation(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("No response received from model after {0} function calls")]
    NoResponse(u32),
}

/// Convenience type alias for Results in this library
pub type Result<T> = std::result::Result<T, AdamastorError>;

// ============ Core Traits ============
pub trait IntoPrompt {
    type Input: Serialize + Default;
    type Output: GeminiSchema + for<'de> Deserialize<'de>;
    fn into_prompt(self) -> Prompt<Self::Input, Self::Output>;
}

/// Trait for types that can be converted into tools
pub trait IntoTool {
    fn into_tool(self) -> Box<dyn GeminiTool>;
}

/// Trait for types that can generate Gemini schemas
pub trait GeminiSchema {
    fn gemini_schema() -> Value;
}

// ============ GeminiSchema implementations for base types ============

impl GeminiSchema for String {
    fn gemini_schema() -> Value {
        json!({"type": "STRING"})
    }
}

impl GeminiSchema for bool {
    fn gemini_schema() -> Value {
        json!({"type": "BOOLEAN"})
    }
}

impl GeminiSchema for i32 {
    fn gemini_schema() -> Value {
        json!({"type": "INTEGER", "format": "int32"})
    }
}

impl GeminiSchema for i64 {
    fn gemini_schema() -> Value {
        json!({"type": "INTEGER", "format": "int64"})
    }
}

impl GeminiSchema for u32 {
    fn gemini_schema() -> Value {
        json!({"type": "INTEGER", "format": "int32"})
    }
}

impl GeminiSchema for u64 {
    fn gemini_schema() -> Value {
        json!({"type": "INTEGER", "format": "int64"})
    }
}

impl GeminiSchema for f32 {
    fn gemini_schema() -> Value {
        json!({"type": "NUMBER", "format": "float"})
    }
}

impl GeminiSchema for f64 {
    fn gemini_schema() -> Value {
        json!({"type": "NUMBER", "format": "double"})
    }
}

impl<T: GeminiSchema> GeminiSchema for Vec<T> {
    fn gemini_schema() -> Value {
        json!({
            "type": "ARRAY",
            "items": T::gemini_schema()
        })
    }
}

impl<T: GeminiSchema> GeminiSchema for Option<T> {
    fn gemini_schema() -> Value {
        let mut schema = T::gemini_schema();
        if let Some(obj) = schema.as_object_mut() {
            obj.insert("nullable".to_string(), json!(true));
        }
        schema
    }
}

impl GeminiSchema for () {
    fn gemini_schema() -> Value {
        json!({"type": "NULL"})
    }
}

impl<V: GeminiSchema> GeminiSchema for HashMap<String, V> {
    fn gemini_schema() -> Value {
        json!({
            "type": "OBJECT",
            "additionalProperties": V::gemini_schema()
        })
    }
}

/// Trait for tools that can be called by Gemini
#[async_trait]
pub trait GeminiTool: Send + Sync {
    fn declaration(&self) -> Value;
    async fn execute(
        &self,
        args: Value,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>>;
    fn name(&self) -> &str;
}

/// A prompt template that transforms Input to Output
pub struct Prompt<Input, Output> {
    template_fn: Box<dyn Fn(&Input) -> String + Send + Sync>,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<Input, Output> Prompt<Input, Output> {
    pub fn new(template_fn: Box<dyn Fn(&Input) -> String + Send + Sync>) -> Self {
        Self {
            template_fn,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn render(&self, input: &Input) -> String {
        (self.template_fn)(input)
    }
}

// ============ File API Support ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHandle {
    pub uri: String,
    pub mime_type: String,
    pub name: Option<String>,
    pub display_name: Option<String>,
}

impl FileHandle {
    fn from_response(response: Value, mime_type: String) -> Result<Self> {
        let file = response.get("file").ok_or_else(|| {
            AdamastorError::ParseError("Missing 'file' field in upload response".to_string())
        })?;

        let uri = file
            .get("uri")
            .and_then(|u| u.as_str())
            .ok_or_else(|| {
                AdamastorError::ParseError("Missing or invalid 'uri' field".to_string())
            })?
            .to_string();

        let name = file
            .get("name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string());

        let display_name = file
            .get("displayName")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());

        Ok(FileHandle {
            uri,
            mime_type,
            name,
            display_name,
        })
    }
}

/// Helper struct for building multipart form data
struct MultipartFormData {
    boundary: String,
    body: Vec<u8>,
}

impl MultipartFormData {
    fn new() -> Self {
        let boundary = format!("----FormBoundary{}", uuid::Uuid::new_v4());
        Self {
            boundary,
            body: Vec::new(),
        }
    }

    fn add_json_field(&mut self, name: &str, value: &Value) {
        let json_str = serde_json::to_string(value).unwrap();
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{}\"\r\n", name).as_bytes(),
        );
        self.body
            .extend_from_slice(b"Content-Type: application/json\r\n\r\n");
        self.body.extend_from_slice(json_str.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    fn add_file_field(&mut self, name: &str, filename: &str, mime_type: &str, data: &[u8]) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
                name, filename
            )
            .as_bytes(),
        );
        self.body
            .extend_from_slice(format!("Content-Type: {}\r\n\r\n", mime_type).as_bytes());
        self.body.extend_from_slice(data);
        self.body.extend_from_slice(b"\r\n");
    }

    fn finish(mut self) -> (String, Vec<u8>) {
        self.body
            .extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        (
            format!("multipart/form-data; boundary={}", self.boundary),
            self.body,
        )
    }
}

// ============ Tool Wrapper ============

pub struct ToolWrapper<F, Input, Output> {
    func: F,
    name: String,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<F, Input, Output, Fut> ToolWrapper<F, Input, Output>
where
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: Future<Output = std::result::Result<Output, Box<dyn std::error::Error + Send + Sync>>>
        + Send,
    Input: GeminiSchema + for<'de> Deserialize<'de> + Send,
    Output: Serialize,
{
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            func,
            name: name.into(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<F, Input, Output, Fut> GeminiTool for ToolWrapper<F, Input, Output>
where
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: Future<Output = std::result::Result<Output, Box<dyn std::error::Error + Send + Sync>>>
        + Send,
    Input: GeminiSchema + for<'de> Deserialize<'de> + Send + Sync,
    Output: Serialize + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn declaration(&self) -> Value {
        let schema = Input::gemini_schema();
        json!({
            "name": self.name,
            "description": format!("Function {}", self.name),
            "parameters": schema
        })
    }

    async fn execute(
        &self,
        args: Value,
    ) -> std::result::Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let input: Input = serde_json::from_value(args)?;
        let result = (self.func)(input).await?;
        Ok(serde_json::to_value(result)?)
    }
}

pub fn tool<F, Input, Output, Fut>(
    name: impl Into<String>,
    func: F,
) -> ToolWrapper<F, Input, Output>
where
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: Future<Output = std::result::Result<Output, Box<dyn std::error::Error + Send + Sync>>>
        + Send,
    Input: GeminiSchema + for<'de> Deserialize<'de> + Send + Sync,
    Output: Serialize + Send + Sync,
{
    ToolWrapper::new(name, func)
}

// ============ Agent Implementation ============

pub struct Agent {
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    persistent_tools: Vec<Arc<dyn GeminiTool>>,
    client: reqwest::Client,
    last_request: RefCell<Instant>,
    requests_per_second: f64,
    max_function_calls: u32,
}

impl Agent {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gemini-2.0-flash-exp".to_string(),
            system_prompt: None,
            persistent_tools: Vec::new(),
            client: reqwest::Client::new(),
            last_request: RefCell::new(Instant::now() - Duration::from_secs(1)),
            requests_per_second: 2.0,
            max_function_calls: 10,
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_requests_per_second(mut self, rps: f64) -> Self {
        self.requests_per_second = rps;
        self
    }

    pub fn with_max_function_calls(mut self, max: u32) -> Self {
        self.max_function_calls = max;
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: IntoTool,
    {
        self.persistent_tools.push(Arc::from(tool.into_tool()));
        self
    }

    async fn wait_if_needed(&self) {
        let min_interval = Duration::from_secs_f64(1.0 / self.requests_per_second);
        let elapsed = self.last_request.borrow().elapsed();

        if elapsed < min_interval {
            tokio::time::sleep(min_interval - elapsed).await;
        }

        *self.last_request.borrow_mut() = Instant::now();
    }

    pub async fn upload_file(
        &self,
        data: &[u8],
        mime_type: impl Into<String>,
    ) -> Result<FileHandle> {
        let mime_type = mime_type.into();
        let display_name = "file";

        self.wait_if_needed().await;

        let mut form = MultipartFormData::new();
        let metadata = json!({
            "file": {
                "displayName": display_name
            }
        });
        form.add_json_field("metadata", &metadata);
        form.add_file_field("file", display_name, &mime_type, data);

        let (content_type, body) = form.finish();

        let url =
            "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart";

        let response = self
            .client
            .post(url)
            .header("X-Goog-Api-Key", &self.api_key)
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AdamastorError::FileOperation(format!(
                "Upload failed: {}",
                error_text
            )));
        }

        let response_json: Value = response.json().await?;
        FileHandle::from_response(response_json, mime_type)
    }

    pub async fn delete_file(&self, file_handle: &FileHandle) -> Result<()> {
        self.wait_if_needed().await;

        let file_id = if let Some(name) = &file_handle.name {
            name.clone()
        } else {
            file_handle
                .uri
                .split('/')
                .last()
                .map(String::from)
                .ok_or_else(|| {
                    AdamastorError::FileOperation("Cannot extract file ID from URI".to_string())
                })?
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}",
            file_id
        );

        let response = self
            .client
            .delete(&url)
            .header("X-Goog-Api-Key", &self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AdamastorError::FileOperation(format!(
                "Delete failed: {}",
                error_text
            )));
        }

        Ok(())
    }

    pub fn prompt<'a, P>(&'a self, prompt_fn: P) -> PromptBuilder<'a, P::Input, P::Output>
    where
        P: IntoPrompt,
    {
        let prompt_data = prompt_fn.into_prompt();
        PromptBuilder {
            agent: self,
            prompt: prompt_data,
            tools: self.persistent_tools.clone(),
            files: Vec::new(),
            retries: 1,
            temperature: None,
            max_tokens: None,
            top_p: None,
            max_function_calls: None,
            _phantom: std::marker::PhantomData,
        }
    }

    async fn call_gemini(&self, request: Value) -> Result<Value> {
        self.wait_if_needed().await;

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model
        );

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AdamastorError::Api(error_text));
        }

        Ok(response.json().await?)
    }
}

// ============ PromptBuilder ============

pub struct PromptBuilder<'a, Input, Output> {
    agent: &'a Agent,
    prompt: Prompt<Input, Output>,
    tools: Vec<Arc<dyn GeminiTool>>,
    files: Vec<FileHandle>,
    retries: u32,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    max_function_calls: Option<u32>,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<'a, Input, Output> PromptBuilder<'a, Input, Output>
where
    Input: Serialize + 'static,
    Output: GeminiSchema + for<'de> Deserialize<'de>,
{
    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: IntoTool,
    {
        self.tools.push(Arc::from(tool.into_tool()));
        self
    }

    pub fn with_file(mut self, file_handle: FileHandle) -> Self {
        self.files.push(file_handle);
        self
    }

    pub fn with_files(mut self, file_handles: Vec<FileHandle>) -> Self {
        self.files.extend(file_handles);
        self
    }

    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn max_function_calls(mut self, max: u32) -> Self {
        self.max_function_calls = Some(max);
        self
    }

    pub async fn invoke<Args>(self, args: Args) -> Result<Output>
    where
        Input: From<Args>,
    {
        // 1. Convert the tuple of arguments into the expected Input struct.
        let input = Input::from(args);

        // 2. The rest of the function proceeds as before, using the `input` variable.
        let rendered_prompt = self.prompt.render(&input);
        let mut request = self.build_request(rendered_prompt);

        let mut last_error = None;
        for attempt in 0..self.retries {
            if attempt > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(500 * (attempt as u64)))
                    .await;
            }

            match self.execute_with_tools(&mut request).await {
                Ok(response_text) => {
                    return serde_json::from_str(&response_text).map_err(|e| {
                        AdamastorError::ParseError(format!("Failed to parse response: {}", e))
                    });
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| AdamastorError::Api("Unknown error".to_string())))
    }

    fn build_request(&self, prompt: String) -> Value {
        let mut contents = vec![];
        let has_tools = !self.tools.is_empty();

        let mut parts = vec![json!({"text": prompt})];

        for file in &self.files {
            parts.push(json!({
                "fileData": {
                    "mimeType": file.mime_type,
                    "fileUri": file.uri
                }
            }));
        }

        contents.push(json!({
            "role": "user",
            "parts": parts
        }));

        let mut request = json!({
            "contents": contents
        });

        if let Some(system_prompt) = &self.agent.system_prompt {
            request["systemInstruction"] = json!({
                "parts": [{"text": system_prompt}]
            });
        }

        if !has_tools {
            let mut generation_config = json!({
                "responseMimeType": "application/json",
                "responseSchema": Output::gemini_schema()
            });

            if let Some(temp) = self.temperature {
                generation_config["temperature"] = json!(temp);
            }
            if let Some(max) = self.max_tokens {
                generation_config["maxOutputTokens"] = json!(max);
            }
            if let Some(p) = self.top_p {
                generation_config["topP"] = json!(p);
            }

            request["generationConfig"] = generation_config;
        } else {
            let mut generation_config = json!({});

            if let Some(temp) = self.temperature {
                generation_config["temperature"] = json!(temp);
            }
            if let Some(max) = self.max_tokens {
                generation_config["maxOutputTokens"] = json!(max);
            }
            if let Some(p) = self.top_p {
                generation_config["topP"] = json!(p);
            }

            if !generation_config.as_object().unwrap().is_empty() {
                request["generationConfig"] = generation_config;
            }

            let declarations: Vec<Value> = self.tools.iter().map(|t| t.declaration()).collect();
            request["tools"] = json!([{
                "functionDeclarations": declarations
            }]);
            request["toolConfig"] = json!({
                "functionCallingConfig": {"mode": "AUTO"}
            });
        }

        request
    }

    async fn execute_with_tools(&self, request: &mut Value) -> Result<String> {
        let mut current_request = request.clone();
        let max_iterations = self
            .max_function_calls
            .unwrap_or(self.agent.max_function_calls);
        let mut final_text_response = None;
        let mut iterations = 0;

        loop {
            if iterations >= max_iterations {
                return Err(AdamastorError::MaxFunctionCalls(max_iterations));
            }
            iterations += 1;

            let response = self.agent.call_gemini(current_request.clone()).await?;

            let first_candidate = response
                .get("candidates")
                .and_then(|c| c.get(0))
                .ok_or_else(|| {
                    AdamastorError::ParseError("Missing candidates in response".to_string())
                })?;

            if let Some(parts) = first_candidate["content"]["parts"].as_array() {
                let mut has_function_call = false;
                let mut function_responses = vec![];

                for part in parts {
                    if let Some(function_call) = part.get("functionCall") {
                        has_function_call = true;
                        let name = function_call["name"].as_str().ok_or_else(|| {
                            AdamastorError::ParseError("Function call missing name".to_string())
                        })?;

                        let default_args = json!({});
                        let args = function_call.get("args").unwrap_or(&default_args);

                        let tool = self
                            .tools
                            .iter()
                            .find(|t| t.name() == name)
                            .ok_or_else(|| AdamastorError::ToolNotFound(name.to_string()))?;

                        match tool.execute(args.clone()).await {
                            Ok(result) => {
                                function_responses.push(json!({
                                    "functionResponse": {
                                        "name": name,
                                        "response": {
                                            "content": result
                                        }
                                    }
                                }));
                            }
                            Err(e) => {
                                // Convert tool errors to our error type
                                return Err(AdamastorError::ToolExecution {
                                    tool: name.to_string(),
                                    error: e.to_string(),
                                });
                            }
                        }
                    } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !has_function_call {
                            final_text_response = Some(text.to_string());
                        }
                    }
                }

                if has_function_call {
                    let mut messages = current_request["contents"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();

                    messages.push(json!({
                        "role": "model",
                        "parts": parts
                    }));

                    messages.push(json!({
                        "role": "user",
                        "parts": function_responses
                    }));

                    current_request["contents"] = json!(messages);
                } else {
                    break;
                }
            }
        }

        if let Some(text) = final_text_response {
            // Phase 2: Format response as JSON
            let format_prompt = format!(
                "Convert the following information into JSON format matching this schema:\n\n\
                Schema: {}\n\n\
                Information to convert:\n{}\n\n\
                Return ONLY valid JSON, no additional text.",
                serde_json::to_string_pretty(&Output::gemini_schema())
                    .map_err(|e| AdamastorError::Json(e))?,
                text
            );

            let format_request = json!({
                "contents": [{
                    "role": "user",
                    "parts": [{"text": format_prompt}]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": Output::gemini_schema(),
                    "temperature": 0.1
                }
            });

            let format_response = self.agent.call_gemini(format_request).await?;

            format_response
                .get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    AdamastorError::ParseError(
                        "Failed to extract formatted JSON from response".to_string(),
                    )
                })
        } else {
            Err(AdamastorError::NoResponse(iterations))
        }
    }
}

// ============ Embedding Support ============

pub struct SingleEmbedBuilder<'a> {
    agent: &'a Agent,
    content: String,
    is_query: bool,
    output_dimensionality: Option<u32>,
}

pub struct BatchEmbedBuilder<'a> {
    agent: &'a Agent,
    contents: Vec<String>,
    is_query: bool,
    output_dimensionality: Option<u32>,
}

impl<'a> SingleEmbedBuilder<'a> {
    fn new(agent: &'a Agent, content: impl Into<String>) -> Self {
        Self {
            agent,
            content: content.into(),
            is_query: false,
            output_dimensionality: None,
        }
    }

    pub fn as_query(mut self) -> Self {
        self.is_query = true;
        self
    }

    pub fn with_dim(mut self, dim: u32) -> Self {
        self.output_dimensionality = Some(dim);
        self
    }

    pub async fn invoke(self) -> Result<Vec<f32>> {
        self.agent.wait_if_needed().await;

        let model = "models/text-embedding-004";
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}:embedContent",
            model
        );

        let mut body = json!({
            "model": model,
            "content": {
                "parts": [{
                    "text": self.content
                }]
            },
            "taskType": if self.is_query { "RETRIEVAL_QUERY" } else { "RETRIEVAL_DOCUMENT" }
        });

        if let Some(dim) = self.output_dimensionality {
            body["outputDimensionality"] = json!(dim);
        }

        let response = self
            .agent
            .client
            .post(&url)
            .header("x-goog-api-key", &self.agent.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AdamastorError::Api(format!(
                "Embedding failed: {}",
                error_text
            )));
        }

        let response_json: Value = response.json().await?;

        response_json
            .get("embedding")
            .and_then(|e| e.get("values"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| AdamastorError::ParseError("Missing embedding values".to_string()))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| AdamastorError::ParseError("Invalid value".to_string()))
            })
            .collect()
    }
}

impl<'a> BatchEmbedBuilder<'a> {
    fn new(agent: &'a Agent, contents: &[impl AsRef<str>]) -> Self {
        Self {
            agent,
            contents: contents.iter().map(|s| s.as_ref().to_string()).collect(),
            is_query: false,
            output_dimensionality: None,
        }
    }

    pub fn as_query(mut self) -> Self {
        self.is_query = true;
        self
    }

    pub fn with_dim(mut self, dim: u32) -> Self {
        self.output_dimensionality = Some(dim);
        self
    }

    pub async fn invoke(self) -> Result<Vec<Vec<f32>>> {
        self.agent.wait_if_needed().await;

        let model = "models/text-embedding-004";
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/{}:batchEmbedContents",
            model
        );

        // Build requests array for batch
        let requests: Vec<Value> = self
            .contents
            .iter()
            .map(|text| {
                let mut req = json!({
                    "model": model,
                    "content": {
                        "parts": [{
                            "text": text
                        }]
                    },
                    "taskType": if self.is_query { "RETRIEVAL_QUERY" } else { "RETRIEVAL_DOCUMENT" }
                });

                if let Some(dim) = self.output_dimensionality {
                    req["outputDimensionality"] = json!(dim);
                }
                req
            })
            .collect();

        let body = json!({
            "requests": requests
        });

        let response = self
            .agent
            .client
            .post(&url)
            .header("x-goog-api-key", &self.agent.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(AdamastorError::Api(format!(
                "Batch embedding failed: {}",
                error_text
            )));
        }

        let response_json: Value = response.json().await?;

        response_json
            .get("embeddings")
            .and_then(|e| e.as_array())
            .ok_or_else(|| AdamastorError::ParseError("Missing embeddings".to_string()))?
            .iter()
            .map(|embedding| {
                embedding
                    .get("values")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| AdamastorError::ParseError("Missing values".to_string()))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .ok_or_else(|| AdamastorError::ParseError("Invalid value".to_string()))
                    })
                    .collect::<Result<Vec<f32>>>()
            })
            .collect()
    }
}

impl Agent {
    pub fn embed(&self, content: impl Into<String>) -> SingleEmbedBuilder {
        SingleEmbedBuilder::new(self, content)
    }

    pub fn embed_batch<S: AsRef<str>>(&self, contents: &[S]) -> BatchEmbedBuilder {
        BatchEmbedBuilder::new(self, contents)
    }
}
