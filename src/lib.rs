use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::any::TypeId;
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use adamastor_macros::schema;

// ============ Error Handling ============

#[derive(Debug, thiserror::Error)]
pub enum AdamastorError {
    #[error("API error: {0}")]
    Api(String),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("File operation failed: {0}")]
    FileOperation(String),
}

pub type Result<T> = std::result::Result<T, AdamastorError>;

// ============ GeminiSchema Trait ============

/// Trait for types that can generate Gemini JSON schemas
pub trait GeminiSchema {
    fn gemini_schema() -> Value;
}

// String is special - it means unstructured text response
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

// ============ Tool Calling ============

type ToolExecutor =
    Box<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

struct ToolDefinition {
    name: String,
    declaration: Value,
    executor: ToolExecutor,
}

// ============ File Handling ============

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

// ============ Rate Limiting ============

struct RateLimiter {
    min_interval: Duration,
    last_request: Mutex<Instant>,
}

impl RateLimiter {
    fn new(requests_per_second: f64) -> Self {
        Self {
            min_interval: Duration::from_secs_f64(1.0 / requests_per_second),
            last_request: Mutex::new(Instant::now() - Duration::from_secs(1)),
        }
    }

    async fn wait(&self) {
        let elapsed = {
            let last = self.last_request.lock().unwrap();
            last.elapsed()
        };

        if elapsed < self.min_interval {
            tokio::time::sleep(self.min_interval - elapsed).await;
        }

        *self.last_request.lock().unwrap() = Instant::now();
    }
}

// ============ Agent ============

/// Main agent for stateless prompt execution
pub struct Agent {
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    client: reqwest::Client,
    rate_limiter: Arc<RateLimiter>,
    max_function_calls: u32,
}

impl Agent {
    /// Create a new stateless agent
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gemini-2.5-flash".to_string(),
            system_prompt: None,
            client: reqwest::Client::new(),
            rate_limiter: Arc::new(RateLimiter::new(2.0)),
            max_function_calls: 10,
        }
    }

    /// Create a new stateful chat agent
    pub fn chat(api_key: impl Into<String>) -> Chat {
        Chat {
            agent: Self::new(api_key),
            history: Vec::new(),
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set a system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set rate limiting
    pub fn with_requests_per_second(mut self, rps: f64) -> Self {
        self.rate_limiter = Arc::new(RateLimiter::new(rps));
        self
    }

    /// Set maximum function call iterations (default: 10)
    pub fn with_max_function_calls(mut self, max: u32) -> Self {
        self.max_function_calls = max;
        self
    }

    /// Create a prompt - returns a builder that can be configured and awaited
    pub fn prompt<T>(&self, text: impl Into<String>) -> PromptBuilder<'_, T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de> + Send + 'static,
    {
        PromptBuilder::new(self, text.into())
    }

    /// Create an encoding task for a single piece of text
    pub fn encode(&self, text: impl Into<String>) -> EncodeBuilder<'_> {
        EncodeBuilder::new(self, text.into())
    }

    /// Create an encoding task for a batch of text
    pub fn encode_batch(&self, texts: Vec<String>) -> EncodeBatchBuilder<'_> {
        EncodeBatchBuilder::new(self, texts)
    }

    /// Upload a file for use in prompts
    pub async fn upload_file(
        &self,
        data: &[u8],
        mime_type: impl Into<String>,
    ) -> Result<FileHandle> {
        let mime_type = mime_type.into();
        let display_name = "file";

        self.rate_limiter.wait().await;

        let mut form = MultipartFormData::new();
        let metadata = json!({"file": {"displayName": display_name}});
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

    async fn call_gemini(&self, request: Value) -> Result<Value> {
        self.rate_limiter.wait().await;

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

    async fn call_embed(&self, request: Value) -> Result<Value> {
        self.rate_limiter.wait().await;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={}",
            self.api_key
        );
        let response = self.client.post(&url).json(&request).send().await?;
        if !response.status().is_success() {
            return Err(AdamastorError::Api(response.text().await?));
        }
        Ok(response.json().await?)
    }

    async fn call_batch_embed(&self, request: Value) -> Result<Value> {
        self.rate_limiter.wait().await;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={}",
            self.api_key
        );
        let response = self.client.post(&url).json(&request).send().await?;
        if !response.status().is_success() {
            return Err(AdamastorError::Api(response.text().await?));
        }
        Ok(response.json().await?)
    }
}

// ============ Chat ============

#[derive(Debug, Clone)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Debug, Clone)]
struct Part {
    text: Option<String>,
    file_data: Option<FileData>,
    function_call: Option<FunctionCall>,
    function_response: Option<FunctionResponse>,
}

#[derive(Debug, Clone)]
struct FileData {
    mime_type: String,
    file_uri: String,
}

#[derive(Debug, Clone)]
struct FunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone)]
struct FunctionResponse {
    name: String,
    response: Value,
}

/// Chat agent that maintains conversation history
pub struct Chat {
    agent: Agent,
    history: Vec<Content>,
}

impl Chat {
    /// Send a message in the conversation - returns a builder that can be configured and awaited
    pub fn send<T>(&mut self, text: impl Into<String>) -> ChatPromptBuilder<'_, T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de> + Send + 'static,
    {
        ChatPromptBuilder::new(self, text.into())
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.agent = self.agent.with_model(model);
        self
    }

    /// Set a system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.agent = self.agent.with_system_prompt(prompt);
        self
    }

    /// Set rate limiting
    pub fn with_requests_per_second(mut self, rps: f64) -> Self {
        self.agent = self.agent.with_requests_per_second(rps);
        self
    }

    /// Set maximum function call iterations (default: 10)
    pub fn with_max_function_calls(mut self, max: u32) -> Self {
        self.agent = self.agent.with_max_function_calls(max);
        self
    }

    /// Get access to the underlying agent for file uploads
    pub fn agent(&self) -> &Agent {
        &self.agent
    }
}

// ============ PromptBuilder ============

/// Builder for configuring and executing a single prompt
pub struct PromptBuilder<'a, T> {
    agent: &'a Agent,
    prompt_text: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    retries: u32,
    files: Vec<FileHandle>,
    tools: Vec<ToolDefinition>,
    max_function_calls: Option<u32>,
    _phantom: PhantomData<T>,
}

impl<'a, T: 'static> PromptBuilder<'a, T> {
    fn new(agent: &'a Agent, prompt_text: String) -> Self {
        Self {
            agent,
            prompt_text,
            temperature: None,
            max_tokens: None,
            top_p: None,
            retries: 1,
            files: Vec::new(),
            tools: Vec::new(),
            max_function_calls: None,
            _phantom: PhantomData,
        }
    }

    /// Set the temperature (0.0 to 1.0)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens in the response
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the top-p value for nucleus sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set the number of retry attempts on failure
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// Attach a file to the prompt
    pub fn with_file(mut self, file: FileHandle) -> Self {
        self.files.push(file);
        self
    }

    /// Add a tool that the model can call
    pub fn with_tool<Args, F, Fut>(mut self, name: impl Into<String>, callback: F) -> Self
    where
        Args: GeminiSchema + for<'de> Deserialize<'de> + Send + 'static,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        let name = name.into();
        let declaration = json!({
            "name": name,
            "description": format!("Tool: {}", name),
            "parameters": Args::gemini_schema()
        });

        let callback = Arc::new(callback);
        let executor: ToolExecutor = Box::new(move |args_json: Value| {
            let callback = Arc::clone(&callback);
            Box::pin(async move {
                let args: Args = serde_json::from_value(args_json).map_err(|e| {
                    AdamastorError::ParseError(format!("Failed to parse tool arguments: {}", e))
                })?;
                callback(args).await
            })
        });

        self.tools.push(ToolDefinition {
            name,
            declaration,
            executor,
        });
        self
    }

    /// Set maximum function call iterations (overrides agent default)
    pub fn with_max_function_calls(mut self, max: u32) -> Self {
        self.max_function_calls = Some(max);
        self
    }

    async fn execute(self) -> Result<T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de>,
    {
        let mut last_error = None;

        for attempt in 0..self.retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }

            match self.execute_once().await {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap())
    }

    async fn execute_once(&self) -> Result<T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de>,
    {
        // Build initial user message
        let mut parts = vec![json!({"text": self.prompt_text})];
        for file in &self.files {
            parts.push(json!({
                "fileData": {
                    "mimeType": file.mime_type,
                    "fileUri": file.uri
                }
            }));
        }

        let mut contents = vec![json!({
            "role": "user",
            "parts": parts
        })];

        let max_iterations = self
            .max_function_calls
            .unwrap_or(self.agent.max_function_calls);

        // Tool calling loop
        for _iteration in 0..max_iterations {
            let request = self.build_request(&contents);
            let response = self.agent.call_gemini(request).await?;

            // Check for function calls (plural - handle parallel calls)
            let function_calls = self.extract_function_calls(&response)?;

            if !function_calls.is_empty() {
                // Add model's response (contains all function calls)
                let model_content = response
                    .get("candidates")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("content"))
                    .ok_or_else(|| {
                        AdamastorError::ParseError("Missing content in response".to_string())
                    })?;
                contents.push(model_content.clone());

                // Execute all tools sequentially and collect responses
                let mut response_parts = Vec::new();
                for function_call in function_calls {
                    // Find the tool
                    let tool = self
                        .tools
                        .iter()
                        .find(|t| t.name == function_call.name)
                        .ok_or_else(|| {
                            AdamastorError::Api(format!("Unknown tool: {}", function_call.name))
                        })?;

                    // Execute the tool
                    let result = (tool.executor)(function_call.args.clone()).await?;

                    // Add to response parts
                    response_parts.push(json!({
                        "functionResponse": {
                            "name": function_call.name,
                            "response": {"result": result}
                        }
                    }));
                }

                // Add all function responses in a single user turn
                contents.push(json!({
                    "role": "user",
                    "parts": response_parts
                }));

                continue;
            }

            // No function calls - parse final response
            return self.parse_response(response);
        }

        Err(AdamastorError::Api(
            "Maximum function call iterations exceeded".to_string(),
        ))
    }

    fn extract_function_calls(&self, response: &Value) -> Result<Vec<FunctionCall>> {
        let parts = response
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array());

        let mut function_calls = Vec::new();

        if let Some(parts) = parts {
            for part in parts {
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .ok_or_else(|| {
                            AdamastorError::ParseError("Missing function name".to_string())
                        })?
                        .to_string();

                    let args = fc.get("args").cloned().unwrap_or(json!({}));

                    function_calls.push(FunctionCall { name, args });
                }
            }
        }

        Ok(function_calls)
    }

    fn build_request(&self, contents: &[Value]) -> Value
    where
        T: GeminiSchema,
    {
        let mut request = json!({
            "contents": contents
        });

        if let Some(system_prompt) = &self.agent.system_prompt {
            request["systemInstruction"] = json!({
                "parts": [{"text": system_prompt}]
            });
        }

        // Add tools if present
        if !self.tools.is_empty() {
            request["tools"] = json!([{
                "functionDeclarations": self.tools.iter()
                    .map(|t| &t.declaration)
                    .collect::<Vec<_>>()
            }]);
        }

        let mut generation_config = json!({});

        // Check if T is String using TypeId - if not, it's a structured type
        if TypeId::of::<T>() != TypeId::of::<String>() {
            generation_config["responseMimeType"] = json!("application/json");
            generation_config["responseSchema"] = T::gemini_schema();
        }

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

        request
    }

    fn parse_response(&self, response: Value) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let text = response
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| AdamastorError::ParseError("Missing text in response".to_string()))?;

        // If T is String, just return the text directly
        if TypeId::of::<T>() == TypeId::of::<String>() {
            // Safe because we checked the type
            let result: T = unsafe { std::mem::transmute_copy(&text.to_string()) };
            std::mem::forget(text.to_string());
            Ok(result)
        } else {
            // Otherwise parse as JSON
            serde_json::from_str(text).map_err(|e| {
                AdamastorError::ParseError(format!(
                    "Failed to parse response into target schema: {}",
                    e
                ))
            })
        }
    }
}

impl<'a, T> IntoFuture for PromptBuilder<'a, T>
where
    T: GeminiSchema + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    type Output = Result<T>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

// ============ ChatPromptBuilder ============

/// Builder for configuring and executing a chat message
pub struct ChatPromptBuilder<'a, T> {
    chat: &'a mut Chat,
    prompt_text: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    retries: u32,
    files: Vec<FileHandle>,
    tools: Vec<ToolDefinition>,
    max_function_calls: Option<u32>,
    _phantom: PhantomData<T>,
}

impl<'a, T: 'static> ChatPromptBuilder<'a, T> {
    fn new(chat: &'a mut Chat, prompt_text: String) -> Self {
        Self {
            chat,
            prompt_text,
            temperature: None,
            max_tokens: None,
            top_p: None,
            retries: 1,
            files: Vec::new(),
            tools: Vec::new(),
            max_function_calls: None,
            _phantom: PhantomData,
        }
    }

    /// Set the temperature (0.0 to 1.0)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens in the response
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the top-p value for nucleus sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set the number of retry attempts on failure
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// Attach a file to this message
    pub fn with_file(mut self, file: FileHandle) -> Self {
        self.files.push(file);
        self
    }

    /// Add a tool that the model can call
    pub fn with_tool<Args, F, Fut>(mut self, name: impl Into<String>, callback: F) -> Self
    where
        Args: GeminiSchema + for<'de> Deserialize<'de> + Send + 'static,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String>> + Send + 'static,
    {
        let name = name.into();
        let declaration = json!({
            "name": name,
            "description": format!("Tool: {}", name),
            "parameters": Args::gemini_schema()
        });

        let callback = Arc::new(callback);
        let executor: ToolExecutor = Box::new(move |args_json: Value| {
            let callback = Arc::clone(&callback);
            Box::pin(async move {
                let args: Args = serde_json::from_value(args_json).map_err(|e| {
                    AdamastorError::ParseError(format!("Failed to parse tool arguments: {}", e))
                })?;
                callback(args).await
            })
        });

        self.tools.push(ToolDefinition {
            name,
            declaration,
            executor,
        });
        self
    }

    /// Set maximum function call iterations (overrides agent default)
    pub fn with_max_function_calls(mut self, max: u32) -> Self {
        self.max_function_calls = Some(max);
        self
    }

    async fn execute(mut self) -> Result<T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de>,
    {
        let mut last_error = None;

        for attempt in 0..self.retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }

            match self.execute_once().await {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }

        Err(last_error.unwrap())
    }

    async fn execute_once(&mut self) -> Result<T>
    where
        T: GeminiSchema + for<'de> Deserialize<'de>,
    {
        // Add user message to history
        let mut user_parts = vec![Part {
            text: Some(self.prompt_text.clone()),
            file_data: None,
            function_call: None,
            function_response: None,
        }];

        for file in &self.files {
            user_parts.push(Part {
                text: None,
                file_data: Some(FileData {
                    mime_type: file.mime_type.clone(),
                    file_uri: file.uri.clone(),
                }),
                function_call: None,
                function_response: None,
            });
        }

        self.chat.history.push(Content {
            role: "user".to_string(),
            parts: user_parts,
        });

        let max_iterations = self
            .max_function_calls
            .unwrap_or(self.chat.agent.max_function_calls);

        // Tool calling loop
        for _iteration in 0..max_iterations {
            let request = self.build_request();
            let response = self.chat.agent.call_gemini(request).await?;

            // Check for function calls (plural - handle parallel calls)
            let function_calls = self.extract_function_calls(&response)?;

            if !function_calls.is_empty() {
                // Add model's function calls to history
                let mut model_parts = Vec::new();
                for function_call in &function_calls {
                    model_parts.push(Part {
                        text: None,
                        file_data: None,
                        function_call: Some(function_call.clone()),
                        function_response: None,
                    });
                }

                self.chat.history.push(Content {
                    role: "model".to_string(),
                    parts: model_parts,
                });

                // Execute all tools sequentially and collect responses
                let mut response_parts = Vec::new();
                for function_call in function_calls {
                    // Find the tool
                    let tool = self
                        .tools
                        .iter()
                        .find(|t| t.name == function_call.name)
                        .ok_or_else(|| {
                            AdamastorError::Api(format!("Unknown tool: {}", function_call.name))
                        })?;

                    // Execute the tool
                    let result = (tool.executor)(function_call.args.clone()).await?;

                    // Add to response parts
                    response_parts.push(Part {
                        text: None,
                        file_data: None,
                        function_call: None,
                        function_response: Some(FunctionResponse {
                            name: function_call.name.clone(),
                            response: json!({"result": result}),
                        }),
                    });
                }

                // Add all function responses to history
                self.chat.history.push(Content {
                    role: "user".to_string(),
                    parts: response_parts,
                });

                continue;
            }

            // No function calls - parse and add to history
            let (result, response_text) = self.parse_response(response)?;

            self.chat.history.push(Content {
                role: "model".to_string(),
                parts: vec![Part {
                    text: Some(response_text),
                    file_data: None,
                    function_call: None,
                    function_response: None,
                }],
            });

            return Ok(result);
        }

        Err(AdamastorError::Api(
            "Maximum function call iterations exceeded".to_string(),
        ))
    }

    fn extract_function_calls(&self, response: &Value) -> Result<Vec<FunctionCall>> {
        let parts = response
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array());

        let mut function_calls = Vec::new();

        if let Some(parts) = parts {
            for part in parts {
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .ok_or_else(|| {
                            AdamastorError::ParseError("Missing function name".to_string())
                        })?
                        .to_string();

                    let args = fc.get("args").cloned().unwrap_or(json!({}));

                    function_calls.push(FunctionCall { name, args });
                }
            }
        }

        Ok(function_calls)
    }

    fn build_request(&self) -> Value
    where
        T: GeminiSchema,
    {
        // Convert history to Gemini format
        let contents: Vec<Value> = self
            .chat
            .history
            .iter()
            .map(|content| {
                let parts: Vec<Value> = content
                    .parts
                    .iter()
                    .map(|part| {
                        if let Some(text) = &part.text {
                            json!({"text": text})
                        } else if let Some(file_data) = &part.file_data {
                            json!({
                                "fileData": {
                                    "mimeType": file_data.mime_type,
                                    "fileUri": file_data.file_uri
                                }
                            })
                        } else if let Some(function_call) = &part.function_call {
                            json!({
                                "functionCall": {
                                    "name": function_call.name,
                                    "args": function_call.args
                                }
                            })
                        } else if let Some(function_response) = &part.function_response {
                            json!({
                                "functionResponse": {
                                    "name": function_response.name,
                                    "response": function_response.response
                                }
                            })
                        } else {
                            json!({"text": ""})
                        }
                    })
                    .collect();

                json!({
                    "role": content.role,
                    "parts": parts
                })
            })
            .collect();

        let mut request = json!({
            "contents": contents
        });

        if let Some(system_prompt) = &self.chat.agent.system_prompt {
            request["systemInstruction"] = json!({
                "parts": [{"text": system_prompt}]
            });
        }

        // Add tools if present
        if !self.tools.is_empty() {
            request["tools"] = json!([{
                "functionDeclarations": self.tools.iter()
                    .map(|t| &t.declaration)
                    .collect::<Vec<_>>()
            }]);
        }

        let mut generation_config = json!({});

        if TypeId::of::<T>() != TypeId::of::<String>() {
            generation_config["responseMimeType"] = json!("application/json");
            generation_config["responseSchema"] = T::gemini_schema();
        }

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

        request
    }

    fn parse_response(&self, response: Value) -> Result<(T, String)>
    where
        T: for<'de> Deserialize<'de>,
    {
        let text = response
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| AdamastorError::ParseError("Missing text in response".to_string()))?;

        let parsed = if TypeId::of::<T>() == TypeId::of::<String>() {
            // Safe because we checked the type
            let result: T = unsafe { std::mem::transmute_copy(&text.to_string()) };
            std::mem::forget(text.to_string());
            result
        } else {
            serde_json::from_str(text).map_err(|e| {
                AdamastorError::ParseError(format!(
                    "Failed to parse response into target schema: {}",
                    e
                ))
            })?
        };

        Ok((parsed, text.to_string()))
    }
}

impl<'a, T> IntoFuture for ChatPromptBuilder<'a, T>
where
    T: GeminiSchema + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    type Output = Result<T>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

// ============ EncodeBuilder ============

/// Builder for configuring and executing a single embedding task
pub struct EncodeBuilder<'a> {
    agent: &'a Agent,
    content: String,
    task_type: Option<String>,
    dimensions: Option<u32>,
}

impl<'a> EncodeBuilder<'a> {
    fn new(agent: &'a Agent, content: String) -> Self {
        Self {
            agent,
            content,
            task_type: None,
            dimensions: None,
        }
    }

    pub fn as_semantic(mut self) -> Self {
        self.task_type = Some("SEMANTIC_SIMILARITY".to_string());
        self
    }

    pub fn as_query(mut self) -> Self {
        self.task_type = Some("RETRIEVAL_QUERY".to_string());
        self
    }

    pub fn as_document(mut self) -> Self {
        self.task_type = Some("RETRIEVAL_DOCUMENT".to_string());
        self
    }

    pub fn as_classification(mut self) -> Self {
        self.task_type = Some("CLASSIFICATION".to_string());
        self
    }

    pub fn dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

impl<'a> IntoFuture for EncodeBuilder<'a> {
    type Output = Result<Vec<f32>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let mut request = json!({
                "model": "models/text-embedding-004",
                "content": {
                    "parts": [{"text": self.content}]
                }
            });

            if let Some(tt) = self.task_type {
                request["taskType"] = json!(tt);
            }
            if let Some(dim) = self.dimensions {
                request["outputDimensionality"] = json!(dim);
            }

            let res = self.agent.call_embed(request).await?;
            let values = res
                .get("embedding")
                .and_then(|e| e.get("values"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    AdamastorError::ParseError("Invalid embedding response".to_string())
                })?;

            let vec = values
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            Ok(vec)
        })
    }
}

// ============ EncodeBatchBuilder ============

/// Builder for configuring and executing batch embedding tasks
pub struct EncodeBatchBuilder<'a> {
    agent: &'a Agent,
    contents: Vec<String>,
    dimensions: Option<u32>,
}

impl<'a> EncodeBatchBuilder<'a> {
    fn new(agent: &'a Agent, contents: Vec<String>) -> Self {
        Self {
            agent,
            contents,
            dimensions: None,
        }
    }

    pub fn dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

impl<'a> IntoFuture for EncodeBatchBuilder<'a> {
    type Output = Result<Vec<Vec<f32>>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let requests: Vec<Value> = self
                .contents
                .into_iter()
                .map(|text| {
                    let mut req = json!({
                        "model": "models/text-embedding-004",
                        "content": { "parts": [{"text": text}] }
                    });
                    if let Some(dim) = self.dimensions {
                        req["outputDimensionality"] = json!(dim);
                    }
                    req
                })
                .collect();

            let request = json!({ "requests": requests });
            let res = self.agent.call_batch_embed(request).await?;

            let embeddings = res
                .get("embeddings")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    AdamastorError::ParseError("Invalid batch embedding response".to_string())
                })?;

            let mut result = Vec::new();
            for emb in embeddings {
                let values = emb
                    .get("values")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        AdamastorError::ParseError("Invalid vector in batch response".to_string())
                    })?;

                result.push(
                    values
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                );
            }

            Ok(result)
        })
    }
}
