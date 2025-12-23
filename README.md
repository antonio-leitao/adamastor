<p align="center">
  <img src='assets/logo.svg' width='200px' align="center"></img>
</p>

<div align="center">
<h3 align="center">Adamastor</h3>
  <p><i>Rust framework for building LLM Agents<br/>
  Type-safe structured outputs and ergonomic API<br/>
  Built for Gemini</i><br/></p>
</div>

<div align="right">
    <i>«Eu sou aquele oculto e grande Cabo<br>
A quem chamais vós outros Tormentório,<br>
    </i></div>

## What is Adamastor?

Adamastor is a Rust framework that makes working with Large Language Models **type-safe**, **ergonomic**, and **reliable**. It leverages Rust's type system to ensure your prompts and responses are validated at compile time, while providing a simple and intuitive API.

## Features

- **Type-safe structured outputs** - Define response schemas with Rust structs
- **Tool calling** - Let models call your functions with type-safe arguments
- **Stateful conversations** - Built-in chat support with automatic history management
- **File handling** - Upload and reference files in prompts (multimodal)
- **Flexible configuration** - Per-request overrides of agent defaults
- **Automatic retries** - Built-in retry logic with exponential backoff
- **Rate limiting** - Configurable request throttling
- **Two modes** - Stateless `Agent` for one-off prompts, `Chat` for conversations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
adamastor = "0.2.0"
```

## Quick Start

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct Poem {
    title: String,
    content: String,
    style: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Structured output - type inference from variable annotation
    let poem: Poem = agent
        .prompt("Write a haiku about Rust")
        .temperature(0.8)
        .await?;

    println!("{}: {}", poem.title, poem.content);

    // Text output - returns String for unstructured responses
    let translation: String = agent
        .prompt("Translate the above poem to Japanese")
        .await?;

    println!("{}", translation);

    Ok(())
}
```

---

## Tutorial: From Simple to Advanced

### 1. Unstructured Text (Simplest)

Get raw text responses without any schema:

```rust
use adamastor::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    let response: String = agent
        .prompt("Explain Rust ownership in one sentence")
        .await?;

    println!("{}", response);
    Ok(())
}
```

**When to use:** Quick answers, content generation where structure doesn't matter.

---

### 2. Structured Output with Type Inference

Define a schema and let Rust infer the type:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
    prep_time_minutes: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Type annotation tells Rust what schema to use
    let recipe: Recipe = agent
        .prompt("Create a pancake recipe")
        .await?;

    println!("Recipe: {}", recipe.name);
    println!("Time: {} minutes", recipe.prep_time_minutes);

    for ingredient in recipe.ingredients {
        println!("- {}", ingredient);
    }

    Ok(())
}
```

**When to use:** When you want structured data and the variable type is clear.

---

### 3. Dynamic Prompts with Functions

Create reusable prompt functions:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
}

// A simple function that returns a prompt string
fn create_recipe_prompt(cuisine: &str, dish_type: &str) -> String {
    format!("Create a traditional {} {} recipe", cuisine, dish_type)
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    let recipe: Recipe = agent
        .prompt(create_recipe_prompt("Italian", "pasta"))
        .temperature(0.7)
        .await?;

    println!("Recipe: {}", recipe.name);
    Ok(())
}
```

**Benefits:**

- ✅ Reusable across your codebase
- ✅ Easy to test
- ✅ Type-safe parameters
- ✅ Clear function signatures

---

### 4. Tool Calling (Function Calling)

Enable models to call your functions to access real-time data or perform actions:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct WeatherArgs {
    /// The city name, e.g., "Tokyo", "London"
    location: String,
}

// Your actual function implementation
async fn get_weather(location: &str) -> Result<String> {
    // In real app: call weather API
    Ok(format!("Weather in {}: Sunny, 22°C", location))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    let response: String = agent
        .prompt("What's the weather like in Tokyo?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            get_weather(&args.location).await
        })
        .await?;

    println!("{}", response);
    // Output: "The weather in Tokyo is sunny with a temperature of 22°C."

    Ok(())
}
```

**How it works:**

1. Model receives your prompt and tool definition
2. Model decides to call `get_weather` with `{"location": "Tokyo"}`
3. Your callback executes and returns the weather data
4. Model generates a natural language response using that data

**Use cases:**

- 🌐 Fetch real-time data (weather, stock prices, news)
- 🗄️ Query databases or APIs
- 📧 Perform actions (send emails, create calendar events)
- 🧮 Run calculations or code execution

---

### 5. Compositional Tool Calling

Models can chain multiple tool calls to accomplish complex tasks:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct LocationArgs {}

#[schema]
struct WeatherArgs {
    location: String,
}

async fn get_current_location() -> Result<String> {
    // In real app: use IP geolocation or GPS
    Ok("San Francisco".to_string())
}

async fn get_weather(location: &str) -> Result<String> {
    Ok(format!("Weather in {}: Sunny, 20°C", location))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Model will first call get_current_location, then get_weather
    let response: String = agent
        .prompt("What's the weather like where I am?")
        .with_tool("get_current_location", |_args: LocationArgs| async {
            get_current_location().await
        })
        .with_tool("get_weather", |args: WeatherArgs| async move {
            get_weather(&args.location).await
        })
        .with_max_function_calls(5)  // Allow multiple tool calls
        .await?;

    println!("{}", response);

    Ok(())
}
```

The model automatically:

1. Calls `get_current_location()` → "San Francisco"
2. Calls `get_weather("San Francisco")` → "Sunny, 20°C"
3. Generates response: "The weather in San Francisco is sunny at 20°C"

---

### 6. Tool Calling with Structured Output

Combine tools with structured responses:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct WeatherArgs {
    location: String,
}

#[schema]
struct TravelPlan {
    destination: String,
    activities: Vec<String>,
    packing_list: Vec<String>,
}

async fn get_weather(location: &str) -> Result<String> {
    Ok(format!("Weather in {}: Rainy, 15°C", location))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Get structured output that uses tool data
    let plan: TravelPlan = agent
        .prompt("Create a weekend travel plan for London")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            get_weather(&args.location).await
        })
        .await?;

    println!("Destination: {}", plan.destination);
    println!("Activities: {:?}", plan.activities);
    println!("Packing: {:?}", plan.packing_list);

    Ok(())
}
```

---

### 7. Turbofish Syntax (Alternative)

When type inference isn't enough, use turbofish:

```rust
let poem = agent
    .prompt("Write a haiku about Rust")
    .await?;  // ❌ Error: can't infer type

// Solution 1: Type annotation
let poem: Poem = agent
    .prompt("Write a haiku about Rust")
    .await?;  // ✅ Works

// Solution 2: Turbofish
let poem = agent
    .prompt::<Poem>("Write a haiku about Rust")
    .await?;  // ✅ Works
```

---

### 8. Stateful Conversations with Chat

Use `Chat` for multi-turn conversations with automatic history management:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct BookIdea {
    title: String,
    logline: String,
    protagonist: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create a stateful chat (requires mut)
    let mut chat = Agent::chat(std::env::var("GEMINI_API_KEY")?);

    // First turn - structured output
    let idea: BookIdea = chat
        .send("Give me a sci-fi book idea about ancient AI")
        .temperature(0.9)
        .await?;

    println!("Title: {}", idea.title);
    println!("Protagonist: {}", idea.protagonist);

    // Second turn - text output (remembers context!)
    let paragraph: String = chat
        .send("Write the opening paragraph of that book")
        .await?;

    println!("\nOpening:\n{}", paragraph);

    // Third turn - still remembers everything
    let analysis: String = chat
        .send("What themes does this story explore?")
        .await?;

    println!("\nThemes:\n{}", analysis);

    Ok(())
}
```

**Key differences:**

- `Agent::chat()` instead of `Agent::new()`
- `.send()` instead of `.prompt()`
- Must be mutable (`mut chat`)
- Automatically maintains conversation history

---

### 9. Tool Calling in Chat

Tools work seamlessly with stateful conversations:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct WeatherArgs {
    location: String,
}

async fn get_weather(location: &str) -> Result<String> {
    Ok(format!("Weather in {}: Clear, 18°C", location))
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut chat = Agent::chat(std::env::var("GEMINI_API_KEY")?);

    // First message with tool
    let response: String = chat
        .send("What's the weather in Paris?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            get_weather(&args.location).await
        })
        .await?;

    println!("{}", response);

    // Follow-up remembers context
    let follow_up: String = chat
        .send("Should I bring an umbrella?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            get_weather(&args.location).await
        })
        .await?;

    println!("{}", follow_up);

    Ok(())
}
```

---

### 10. Manual Chaining (Type-Safe)

Chain prompts by passing outputs as inputs:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct Article {
    title: String,
    content: String,
    word_count: u32,
}

#[schema]
struct Summary {
    main_points: Vec<String>,
    one_liner: String,
}

#[schema]
struct Quiz {
    questions: Vec<String>,
    difficulty: String,
}

fn write_article(topic: &str) -> String {
    format!("Write a detailed article about {}", topic)
}

fn summarize(article: &Article) -> String {
    format!("Summarize this article:\n\nTitle: {}\n\n{}",
        article.title, article.content)
}

fn create_quiz(summary: &Summary) -> String {
    format!("Create a quiz based on these main points: {:?}",
        summary.main_points)
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Step 1: Write article
    let article: Article = agent
        .prompt(write_article("Rust async programming"))
        .await?;

    println!("✓ Article written: {}", article.title);

    // Step 2: Summarize (uses article from step 1)
    let summary: Summary = agent
        .prompt(summarize(&article))
        .await?;

    println!("✓ Summary: {}", summary.one_liner);

    // Step 3: Create quiz (uses summary from step 2)
    let quiz: Quiz = agent
        .prompt(create_quiz(&summary))
        .await?;

    println!("✓ Quiz created with {} questions", quiz.questions.len());

    Ok(())
}
```

**Why manual chaining?**

- Explicit and easy to understand
- Inspect intermediate results
- Full control over error handling
- Debug each step independently

---

## 11. Text Embeddings (Encoding)

Convert text into numerical vectors for semantic search, clustering, or classification using `models/text-embedding-004`.

### Single Encoding

```rust
use adamastor::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Default semantic embedding (768 dimensions)
    let vector: Vec<f32> = agent
        .encode("Adamastor is a Rust framework")
        .await?;

    // Specialized query embedding with custom dimensions
    let query_vec: Vec<f32> = agent
        .encode("What is Adamastor?")
        .as_query()         // Optimized for retrieval
        .dimensions(512)    // Truncate output vector
        .await?;

    println!("Vector length: {}", query_vec.len());
    Ok(())
}
```

### Batch Encoding

For high-performance processing, use the batch API to encode multiple strings in a **single HTTP request**.

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    let texts = vec![
        "Rust is fast".to_string(),
        "Python is easy".to_string(),
        "Go is simple".to_string(),
    ];

    let matrix: Vec<Vec<f32>> = agent
        .encode_batch(texts)
        .dimensions(768)
        .await?;

    println!("Encoded {} strings", matrix.len());
    Ok(())
}
```

**Encoding Methods:**

| Method                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `.as_query()`          | Optimized for the query side of a search (Short questions)   |
| `.as_document()`       | Optimized for the document side of a search (Chunks of text) |
| `.as_classification()` | Optimized for use as input to a classifier                   |
| `.dimensions(u32)`     | Sets the output vector size (clipping). Defaults to 768.     |

**Note:** Unlike prompts, the encoding API is hardcoded to use `text-embedding-004` to ensure the most modern and efficient vector representation.

---

## Configuration

### Agent Configuration (Global Defaults)

Configure default settings for all prompts:

```rust
use adamastor::Agent;

// Stateless agent
let agent = Agent::new(api_key)
    .with_model("gemini-2.0-flash")                    // Model selection
    .with_system_prompt("You are a helpful assistant") // System instructions
    .with_requests_per_second(3.0)                     // Rate limiting (default: 2.0)
    .with_max_function_calls(10);                      // Tool call limit (default: 10)

// Stateful chat agent
let mut chat = Agent::chat(api_key)
    .with_model("gemini-2.5-pro")
    .with_system_prompt("You are a creative writing assistant")
    .with_requests_per_second(1.5)
    .with_max_function_calls(5);
```

---

### Prompt Configuration (Per-Request Overrides)

Override agent defaults for individual requests:

```rust
let response: String = agent
    .prompt("Tell me a story")

    // Generation parameters
    .temperature(0.9)          // Creativity (0.0 - 1.0, default: uses model default)
    .max_tokens(2048)          // Maximum response length (default: no limit)
    .top_p(0.95)              // Nucleus sampling (0.0 - 1.0, default: uses model default)

    // Reliability
    .retries(3)               // Retry on failure (default: 1)

    // Files
    .with_file(file_handle)   // Attach a single file

    // Tools
    .with_tool("tool_name", callback)  // Add a tool
    .with_max_function_calls(5)        // Override tool call limit

    .await?;
```

**All Configuration Methods:**

| Method                          | Type         | Description                                 | Default       |
| ------------------------------- | ------------ | ------------------------------------------- | ------------- |
| `.temperature(f32)`             | `0.0 - 1.0`  | Controls randomness. Higher = more creative | Model default |
| `.max_tokens(u32)`              | `u32`        | Maximum tokens in response                  | No limit      |
| `.top_p(f32)`                   | `0.0 - 1.0`  | Nucleus sampling threshold                  | Model default |
| `.retries(u32)`                 | `u32`        | Number of retry attempts on failure         | `1`           |
| `.with_file(FileHandle)`        | `FileHandle` | Attach a file to the prompt                 | None          |
| `.with_tool(name, callback)`    | Closure      | Add a tool the model can call               | None          |
| `.with_max_function_calls(u32)` | `u32`        | Maximum tool call iterations                | `10`          |

---

### EncodeBuilder / BatchEncodeBuilder Methods

```rust
// Configuration
.as_query() -> Self
.as_document() -> Self
.as_classification() -> Self
.dimensions(dims: u32) -> Self

// Execution
async .await -> Result<Vec<f32>>       // For EncodeBuilder
async .await -> Result<Vec<Vec<f32>>> // For BatchEncodeBuilder
```

## Working with Files (Multimodal)

Upload and reference files in prompts:

```rust
use adamastor::{Agent, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?);

    // Upload a file
    let file_data = std::fs::read("document.pdf")?;
    let file_handle = agent
        .upload_file(&file_data, "application/pdf")
        .await?;

    // Use in prompt
    let summary: String = agent
        .prompt("Summarize the key findings from this document")
        .with_file(file_handle.clone())
        .await?;

    println!("Summary:\n{}", summary);

    Ok(())
}
```

**Supported File Types:**

- PDFs: `application/pdf`
- Images: `image/jpeg`, `image/png`, `image/webp`
- Text: `text/plain`, `text/markdown`
- And more - see [Gemini docs](https://ai.google.dev/gemini-api/docs/prompting_with_media)

**For Chat:**

```rust
let mut chat = Agent::chat(api_key);

// Upload via the underlying agent
let file_handle = chat.agent().upload_file(&file_data, "image/jpeg").await?;

// Use in conversation
let description: String = chat
    .send("What's in this image?")
    .with_file(file_handle)
    .await?;
```

**Note:** Gemini automatically deletes uploaded files after 48 hours.

---

## Schema Definition

The `#[schema]` macro generates everything you need for type-safe structured outputs:

```rust
use adamastor::schema;

#[schema]
struct BlogPost {
    /// The title of the blog post (doc comments become field descriptions!)
    title: String,

    /// Main content in markdown format
    content: String,

    /// Tags for categorization
    tags: Vec<String>,

    /// Estimated reading time in minutes
    reading_time: u32,

    /// Optional author name
    author: Option<String>,
}
```

**What `#[schema]` generates:**

- ✅ `Serialize` and `Deserialize` (serde)
- ✅ `GeminiSchema` trait (converts to JSON Schema for API)
- ✅ `Debug`, `Clone`, and `Default` traits
- ✅ Converts doc comments to field descriptions in the schema

---

### Supported Types

The `#[schema]` macro works with:

**Primitives:**

- `String`
- `bool`
- `i32`, `i64`, `u32`, `u64`
- `f32`, `f64`

**Collections:**

- `Vec<T>` where T is any supported type

**Optional:**

- `Option<T>` where T is any supported type

**Nested:**

```rust
#[schema]
struct Author {
    name: String,
    bio: String,
}

#[schema]
struct Article {
    title: String,
    author: Author,        // ✅ Nested struct
    co_authors: Vec<Author>, // ✅ Vec of structs
}
```

---

### Tool Arguments

Tool arguments use the same `#[schema]` macro:

```rust
use adamastor::schema;

#[schema]
struct SearchArgs {
    /// The search query string
    query: String,

    /// Maximum number of results to return
    max_results: u32,

    /// Optional category filter
    category: Option<String>,
}

// Use in tool
agent.prompt("Search for Rust tutorials")
    .with_tool("search", |args: SearchArgs| async move {
        perform_search(&args.query, args.max_results).await
    })
```

---

## Error Handling

Adamastor provides comprehensive error types:

```rust
use adamastor::{AdamastorError, Result};

match agent.prompt("Hello").await {
    Ok(response) => println!("Success: {}", response),

    Err(AdamastorError::Api(msg)) => {
        eprintln!("API error: {}", msg);
        // Gemini API returned an error
    }

    Err(AdamastorError::Network(e)) => {
        eprintln!("Network error: {}", e);
        // Connection failed, timeout, etc.
    }

    Err(AdamastorError::ParseError(msg)) => {
        eprintln!("Parse error: {}", msg);
        // Response didn't match expected schema
    }

    Err(AdamastorError::Json(e)) => {
        eprintln!("JSON error: {}", e);
        // Invalid JSON in response
    }

    Err(AdamastorError::FileOperation(msg)) => {
        eprintln!("File error: {}", msg);
        // File upload or deletion failed
    }
}
```

**Error Types:**

| Error                     | Description             | Common Causes                                    |
| ------------------------- | ----------------------- | ------------------------------------------------ |
| `Api(String)`             | Gemini API error        | Invalid API key, quota exceeded, invalid request |
| `Network(reqwest::Error)` | Network failure         | No internet, timeout, DNS issues                 |
| `ParseError(String)`      | Response parsing failed | Response doesn't match schema, malformed output  |
| `Json(serde_json::Error)` | JSON error              | Invalid JSON structure                           |
| `FileOperation(String)`   | File operation failed   | Upload failed, file not found, invalid MIME type |

**Tool Errors:**

If a tool callback returns an error, it's immediately propagated to the caller:

```rust
let result: String = agent
    .prompt("Get data")
    .with_tool("fetch_data", |args: DataArgs| async move {
        // If this returns Err, the entire prompt fails
        fetch_from_database(&args.id).await
    })
    .await?;  // Error propagated here
```

---

## API Reference

### Agent Methods

```rust
// Creation
Agent::new(api_key: impl Into<String>) -> Agent
Agent::chat(api_key: impl Into<String>) -> Chat

// Configuration (chainable)
.with_model(model: impl Into<String>) -> Self
.with_system_prompt(prompt: impl Into<String>) -> Self
.with_requests_per_second(rps: f64) -> Self
.with_max_function_calls(max: u32) -> Self
.encode(text: impl Into<String>) -> EncodeBuilder<'_>
.encode_batch(texts: Vec<String>) -> BatchEncodeBuilder<'_>


// Prompt execution
.prompt<T>(text: impl Into<String>) -> PromptBuilder<'_, T>
    where T: GeminiSchema + Deserialize + Send + 'static

// File operations
async .upload_file(data: &[u8], mime_type: impl Into<String>) -> Result<FileHandle>
```

### Chat Methods

```rust
// Sending messages
.send<T>(text: impl Into<String>) -> ChatPromptBuilder<'_, T>
    where T: GeminiSchema + Deserialize + Send + 'static

// Configuration (same as Agent)
.with_model(model: impl Into<String>) -> Self
.with_system_prompt(prompt: impl Into<String>) -> Self
.with_requests_per_second(rps: f64) -> Self
.with_max_function_calls(max: u32) -> Self

// Access underlying agent
.agent() -> &Agent
```

### PromptBuilder / ChatPromptBuilder Methods

```rust
// Configuration (chainable)
.temperature(temp: f32) -> Self           // 0.0 - 1.0
.max_tokens(tokens: u32) -> Self
.top_p(p: f32) -> Self                    // 0.0 - 1.0
.retries(n: u32) -> Self
.with_file(file: FileHandle) -> Self
.with_tool<Args, F, Fut>(name: impl Into<String>, callback: F) -> Self
    where Args: GeminiSchema + Deserialize + Send + 'static,
          F: Fn(Args) -> Fut + Send + Sync + 'static,
          Fut: Future<Output = Result<String>> + Send + 'static
.with_max_function_calls(max: u32) -> Self

// Execution (consumes builder)
async .await -> Result<T>
```

---

## Best Practices

### 1. Use Type Annotations for Clarity

```rust
// ✅ Good - clear what type is expected
let recipe: Recipe = agent.prompt("Create a recipe").await?;

// ❌ Less clear - compiler has to infer
let recipe = agent.prompt::<Recipe>("Create a recipe").await?;
```

### 2. Create Helper Functions for Reusable Prompts

```rust
// ✅ Good - reusable and testable
fn translate_to(text: &str, language: &str) -> String {
    format!("Translate '{}' to {}", text, language)
}

let spanish = agent.prompt(translate_to("Hello", "Spanish")).await?;
let french = agent.prompt(translate_to("Hello", "French")).await?;
```

### 3. Use Chat for Multi-Turn Conversations

```rust
// ✅ Good - stateful, remembers context
let mut chat = Agent::chat(api_key);
let first: String = chat.send("What is Rust?").await?;
let follow_up: String = chat.send("What are its benefits?").await?; // Remembers context

// ❌ Bad - stateless, loses context
let agent = Agent::new(api_key);
let first: String = agent.prompt("What is Rust?").await?;
let follow_up: String = agent.prompt("What are its benefits?").await?; // No context!
```

### 4. Add Doc Comments to Schema Fields

```rust
// ✅ Good - helps the model understand what you want
#[schema]
struct Analysis {
    /// A score from 0-100 indicating quality
    quality_score: u32,

    /// List of specific issues found, one per line
    issues: Vec<String>,
}

// Doc comments are especially important for tool arguments
#[schema]
struct WeatherArgs {
    /// The city name, e.g., "San Francisco", "London", "Tokyo"
    location: String,
}
```

### 5. Keep Tool Callbacks Simple

```rust
// ✅ Good - simple, focused tool
.with_tool("get_weather", |args: WeatherArgs| async move {
    get_weather(&args.location).await
})

// ❌ Bad - complex logic in callback
.with_tool("complex_tool", |args: ComplexArgs| async move {
    let data1 = fetch_data1().await?;
    let data2 = fetch_data2().await?;
    let processed = process(data1, data2)?;
    // ... many more lines
    Ok(result)
})

// ✅ Better - extract to function
async fn complex_tool_impl(args: ComplexArgs) -> Result<String> {
    // Complex logic here
}

.with_tool("complex_tool", complex_tool_impl)
```

### 6. Handle Errors Appropriately

```rust
// ✅ Good - specific error handling
match agent.prompt("Hello").await {
    Ok(response) => process(response),
    Err(AdamastorError::ParseError(_)) => {
        // Maybe retry with a different prompt
    }
    Err(e) => return Err(e),
}

// ❌ Bad - swallowing errors
let response = agent.prompt("Hello").await.unwrap_or_default();
```

### 7. Configure Tool Limits Appropriately

```rust
// For simple tasks - low limit
let response: String = agent
    .prompt("What's 2+2?")
    .with_tool("calculate", calculator_tool)
    .with_max_function_calls(2)  // Only need 1-2 calls
    .await?;

// For complex workflows - higher limit
let response: String = agent
    .prompt("Research and summarize the latest AI news")
    .with_tool("search", search_tool)
    .with_tool("fetch_article", fetch_tool)
    .with_max_function_calls(15)  // May need many calls
    .await?;
```

---

## Examples

Check the `examples/` directory for more:

- `examples/ux.rs` - Comprehensive usage example with tool calling
- Run with: `cargo run --example ux`

Named after the mythical giant **Adamastor** from Portuguese literature (_Os Lusíadas_ by Luís de Camões), who guards the Cape of Good Hope
