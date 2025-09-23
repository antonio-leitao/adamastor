use async_trait::async_trait;
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Re-export macros from artisan-macros
pub use artisan_macros::{prompt, schema, tool};

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

/// Trait for tools that can be called by Gemini
#[async_trait]
pub trait GeminiTool: Send + Sync {
    fn declaration(&self) -> Value;
    async fn execute(&self, args: Value)
    -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
    fn name(&self) -> &str;
}

/// A prompt template that transforms Input to Output
pub struct Prompt<Input, Output> {
    template: String,
    input_name: String,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<Input, Output> Prompt<Input, Output> {
    pub fn new(template: &'static str, input_name: &'static str) -> Self {
        Self {
            template: template.to_string(),
            input_name: input_name.to_string(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn template(&self) -> &str {
        &self.template
    }
    pub fn input_name(&self) -> &str {
        &self.input_name
    }
}

// ============ File API Support ============

/// Represents an uploaded file in Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHandle {
    /// The URI of the uploaded file (e.g., "https://generativelanguage.googleapis.com/v1beta/files/...")
    pub uri: String,
    /// The MIME type of the file
    pub mime_type: String,
    /// The name of the file (optional)
    pub name: Option<String>,
    /// The display name for the file (optional)
    pub display_name: Option<String>,
}

impl FileHandle {
    /// Create a FileHandle from a Gemini file response
    fn from_response(
        response: Value,
        mime_type: String,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = response
            .get("file")
            .ok_or("Missing 'file' field in upload response")?;

        let uri = file
            .get("uri")
            .and_then(|u| u.as_str())
            .ok_or("Missing or invalid 'uri' field")?
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

/// Wrapper that converts async functions into GeminiTool trait implementations
pub struct ToolWrapper<F, Input, Output> {
    func: F,
    name: String,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<F, Input, Output, Fut> ToolWrapper<F, Input, Output>
where
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Output, Box<dyn std::error::Error + Send + Sync>>> + Send,
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
    Fut: Future<Output = Result<Output, Box<dyn std::error::Error + Send + Sync>>> + Send,
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
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let input: Input = serde_json::from_value(args)?;
        let result = (self.func)(input).await?;
        Ok(serde_json::to_value(result)?)
    }
}

// Helper function to create tool wrappers more easily
pub fn tool<F, Input, Output, Fut>(
    name: impl Into<String>,
    func: F,
) -> ToolWrapper<F, Input, Output>
where
    F: Fn(Input) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Output, Box<dyn std::error::Error + Send + Sync>>> + Send,
    Input: GeminiSchema + for<'de> Deserialize<'de> + Send + Sync,
    Output: Serialize + Send + Sync,
{
    ToolWrapper::new(name, func)
}

// ============ Agent Implementation ============

/// The main agent that orchestrates interactions with Gemini
pub struct Agent {
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    persistent_tools: Vec<Arc<dyn GeminiTool>>,
    client: reqwest::Client,
    last_request: RefCell<Instant>,
    requests_per_second: f64,
}

impl Agent {
    /// Create a new agent with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gemini-2.0-flash-exp".to_string(),
            system_prompt: None,
            persistent_tools: Vec::new(),
            client: reqwest::Client::new(),
            last_request: RefCell::new(Instant::now() - Duration::from_secs(1)),
            requests_per_second: 2.0,
        }
    }

    /// Set a system prompt that will be used for all interactions
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the rate limit for API requests (default: 2 requests per second)
    pub fn with_requests_per_second(mut self, rps: f64) -> Self {
        self.requests_per_second = rps;
        self
    }

    /// Add a persistent tool that will be available for all prompts
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

    /// Upload a file to Gemini
    pub async fn upload_file(
        &self,
        data: &[u8],
        mime_type: impl Into<String>,
    ) -> Result<FileHandle, Box<dyn std::error::Error + Send + Sync>> {
        let mime_type = mime_type.into();
        let display_name = "file";

        // Rate limit the request
        self.wait_if_needed().await;

        // Build multipart form data
        let mut form = MultipartFormData::new();

        // Add metadata
        let metadata = json!({
            "file": {
                "displayName": display_name
            }
        });
        form.add_json_field("metadata", &metadata);

        // Add file data
        form.add_file_field("file", display_name, &mime_type, data);

        let (content_type, body) = form.finish();

        // Make the upload request
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
            return Err(format!("File upload failed: {}", error_text).into());
        }

        let response_json: Value = response.json().await?;
        FileHandle::from_response(response_json, mime_type)
    }

    /// Delete an uploaded file
    pub async fn delete_file(
        &self,
        file_handle: &FileHandle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.wait_if_needed().await;

        // Correctly and safely extract the file ID
        let file_id = if let Some(name) = &file_handle.name {
            name.clone()
        } else {
            file_handle
                .uri
                .split('/')
                .last()
                .map(String::from)
                .ok_or("Cannot extract file name from FileHandle URI")?
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
            return Err(format!("Failed to delete file: {}", error_text).into());
        }

        Ok(())
    }

    /// Start building a prompt execution.
    /// This method works for prompts defined with or without input arguments.
    pub fn prompt<'a, P>(&'a self, prompt_fn: P) -> PromptBuilder<'a, P::Input, P::Output>
    where
        P: IntoPrompt,
    {
        let prompt_data = prompt_fn.into_prompt();
        PromptBuilder {
            agent: self,
            prompt_template: prompt_data.template().to_string(),
            input_name: prompt_data.input_name().to_string(),
            tools: self.persistent_tools.clone(),
            files: Vec::new(),
            retries: 1,
            temperature: None,
            max_tokens: None,
            top_p: None,
            _phantom: std::marker::PhantomData,
        }
    }

    async fn call_gemini(
        &self,
        request: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        // Rate limit the request
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
            return Err(format!("API error: {}", error_text).into());
        }

        Ok(response.json().await?)
    }
}

// ============ PromptBuilder - Fluent API ============

/// Builder for configuring and executing prompts
pub struct PromptBuilder<'a, Input, Output> {
    agent: &'a Agent,
    prompt_template: String,
    input_name: String,
    tools: Vec<Arc<dyn GeminiTool>>,
    files: Vec<FileHandle>,
    retries: u32,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    _phantom: std::marker::PhantomData<(Input, Output)>,
}

impl<'a, Input, Output> PromptBuilder<'a, Input, Output>
where
    Input: Serialize + 'static,
    Output: GeminiSchema + for<'de> Deserialize<'de>,
{
    /// Add a tool to be available for this specific prompt
    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: IntoTool,
    {
        self.tools.push(Arc::from(tool.into_tool()));
        self
    }

    /// Add a file to be included in this prompt
    pub fn with_file(mut self, file_handle: FileHandle) -> Self {
        self.files.push(file_handle);
        self
    }

    /// Add multiple files to be included in this prompt
    pub fn with_files(mut self, file_handles: Vec<FileHandle>) -> Self {
        self.files.extend(file_handles);
        self
    }

    /// Set the number of retries on failure (default: 1)
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// Set the temperature for response generation (0.0 to 1.0)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens to generate
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the top_p parameter for nucleus sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Execute the prompt with the given input
    pub async fn invoke(
        self,
        input: Input,
    ) -> Result<Output, Box<dyn std::error::Error + Send + Sync>> {
        // Render the prompt template with input
        let rendered_prompt = if std::any::TypeId::of::<Input>() == std::any::TypeId::of::<()>() {
            // For empty input (unit type), use the template as-is
            self.prompt_template.clone()
        } else {
            // For regular input, render with Handlebars
            let mut handlebars = Handlebars::new();
            handlebars.register_escape_fn(handlebars::no_escape);

            let input_json = serde_json::to_value(&input)?;
            handlebars.render_template(
                &self.prompt_template,
                &serde_json::json!({ &self.input_name: input_json }),
            )?
        };

        // Build the request
        let mut request = self.build_request(rendered_prompt);

        // Try with retries
        let mut last_error = None;
        for attempt in 0..self.retries {
            if attempt > 0 {
                // Exponential backoff
                tokio::time::sleep(tokio::time::Duration::from_millis(500 * (attempt as u64)))
                    .await;
            }

            match self.execute_with_tools(&mut request).await {
                Ok(response_text) => {
                    // Parse the response as Output type
                    return Ok(serde_json::from_str(&response_text)?);
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| "Unknown error".into()))
    }

    fn build_request(&self, prompt: String) -> Value {
        let mut contents = vec![];

        // Determine if we're using tools
        let has_tools = !self.tools.is_empty();

        // Build the user prompt
        let user_prompt = prompt;

        // Build parts for the user message
        let mut parts = vec![json!({"text": user_prompt})];

        // Add file parts if any
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
        // Only use structured output when NOT using tools
        if !has_tools {
            let mut generation_config = json!({
                "responseMimeType": "application/json",
                "responseSchema": Output::gemini_schema()
            });

            // Add optional parameters
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
            // When using tools, we can't use JSON response mode
            // But we can still set other generation parameters
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

            // Add tool declarations
            let declarations: Vec<Value> = self.tools.iter().map(|t| t.declaration()).collect();
            // Just regular tool setup, no JSON instructions
            request["tools"] = json!([{
                "functionDeclarations": declarations
            }]);
            request["toolConfig"] = json!({
                "functionCallingConfig": {"mode": "AUTO"}
            });
        }

        request
    }

    async fn execute_with_tools(
        &self,
        request: &mut Value,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut current_request = request.clone();
        let max_iterations = 10;
        let mut final_text_response = None;

        // Phase 1: Execute tool calls and get the natural language response
        for _ in 0..max_iterations {
            let response = self.agent.call_gemini(current_request.clone()).await?;

            let Some(first_candidate) = response.get("candidates").and_then(|c| c.get(0)) else {
                return Err("Invalid response format: missing candidates".into());
            };

            // Check for function calls
            if let Some(parts) = first_candidate["content"]["parts"].as_array() {
                let mut has_function_call = false;
                let mut function_responses = vec![];

                for part in parts {
                    if let Some(function_call) = part.get("functionCall") {
                        has_function_call = true;
                        // ... execute function (same as your existing code) ...
                        let name = function_call["name"]
                            .as_str()
                            .ok_or("Function call missing name")?;

                        let default_args = json!({});
                        let args = function_call.get("args").unwrap_or(&default_args);

                        let tool = self
                            .tools
                            .iter()
                            .find(|t| t.name() == name)
                            .ok_or_else(|| format!("Unknown tool: {}", name))?;

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
                                function_responses.push(json!({
                                    "functionResponse": {
                                        "name": name,
                                        "response": {
                                            "content": json!({
                                                "error": e.to_string()
                                            })
                                        }
                                    }
                                }));
                            }
                        }
                    } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        // Capture any text response
                        if !has_function_call {
                            final_text_response = Some(text.to_string());
                        }
                    }
                }

                if has_function_call {
                    // Continue the conversation with function results
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
                    // No more function calls, we have our final response
                    break;
                }
            }
        }

        // Phase 2: Transform the text response into structured output
        // This happens transparently - the user doesn't know about this second call
        if let Some(text) = final_text_response {
            // Build a new request specifically for formatting
            let format_prompt = format!(
                "Convert the following information into JSON format matching this schema:\n\n\
            Schema: {}\n\n\
            Information to convert:\n{}\n\n\
            Return ONLY valid JSON, no additional text.",
                serde_json::to_string_pretty(&Output::gemini_schema()).unwrap(),
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
                    "temperature": 0.1  // Low temperature for consistent formatting
                }
            });

            // Make the formatting call
            let format_response = self.agent.call_gemini(format_request).await?;

            if let Some(formatted_text) = format_response
                .get("candidates")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.get(0))
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
            {
                return Ok(formatted_text.to_string());
            } else {
                return Err("Failed to format response as JSON".into());
            }
        }

        Err("No final response received from model".into())
    }
}
