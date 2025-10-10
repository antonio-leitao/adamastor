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

### 4. Turbofish Syntax (Alternative)

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

### 5. Stateful Conversations with Chat

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

### 6. Manual Chaining (Type-Safe)

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

## Configuration

### Agent Configuration (Global Defaults)

Configure default settings for all prompts:

```rust
use adamastor::Agent;

// Stateless agent
let agent = Agent::new(api_key)
    .with_model("gemini-2.0-flash")                    // Model selection
    .with_system_prompt("You are a helpful assistant") // System instructions
    .with_requests_per_second(3.0);                    // Rate limiting (default: 2.0)

// Stateful chat agent
let mut chat = Agent::chat(api_key)
    .with_model("gemini-2.5-pro")
    .with_system_prompt("You are a creative writing assistant")
    .with_requests_per_second(1.5);
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

    .await?;
```

**All Configuration Methods:**

| Method                   | Type         | Description                                 | Default       |
| ------------------------ | ------------ | ------------------------------------------- | ------------- |
| `.temperature(f32)`      | `0.0 - 1.0`  | Controls randomness. Higher = more creative | Model default |
| `.max_tokens(u32)`       | `u32`        | Maximum tokens in response                  | No limit      |
| `.top_p(f32)`            | `0.0 - 1.0`  | Nucleus sampling threshold                  | Model default |
| `.retries(u32)`          | `u32`        | Number of retry attempts on failure         | `1`           |
| `.with_file(FileHandle)` | `FileHandle` | Attach a file to the prompt                 | None          |

---

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

    // Optional: Delete file (auto-deleted after 48h anyway)
    agent.delete_file(&file_handle).await?;

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

### Advanced Schema Example

```rust
use adamastor::schema;

#[schema]
struct CodeAnalysis {
    /// The programming language detected
    language: String,

    /// Overall code quality score (0-100)
    quality_score: u32,

    /// List of identified issues
    issues: Vec<Issue>,

    /// Suggested improvements
    suggestions: Vec<String>,

    /// Whether the code follows best practices
    follows_best_practices: bool,

    /// Estimated complexity (simple, moderate, complex)
    complexity: String,

    /// Optional security concerns
    security_notes: Option<String>,
}

#[schema]
struct Issue {
    /// Line number where issue occurs
    line: u32,

    /// Severity: "error", "warning", or "info"
    severity: String,

    /// Description of the issue
    message: String,
}

// Use it
let analysis: CodeAnalysis = agent
    .prompt("Analyze this Rust code: fn main() { ... }")
    .await?;

println!("Quality: {}/100", analysis.quality_score);
for issue in analysis.issues {
    println!("[{}] Line {}: {}", issue.severity, issue.line, issue.message);
}
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

---

## Complete Real-World Example

Here's a complete example showing multiple features:

```rust
use adamastor::{Agent, schema, Result};

#[schema]
struct Character {
    name: String,
    age: u32,
    personality_traits: Vec<String>,
    backstory: String,
}

#[schema]
struct Scene {
    setting: String,
    dialogue: Vec<String>,
    action: String,
}

fn create_character(genre: &str) -> String {
    format!("Create a compelling {} character with depth and motivation", genre)
}

fn write_scene(character: &Character, situation: &str) -> String {
    format!(
        "Write a scene where {} (personality: {:?}) faces this situation: {}",
        character.name, character.personality_traits, situation
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configure agent
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?)
        .with_model("gemini-1.5-flash")
        .with_system_prompt("You are a creative writing assistant specializing in fiction")
        .with_requests_per_second(2.0);

    println!("🎭 Character Creator\n");

    // Step 1: Create a character
    let character: Character = agent
        .prompt(create_character("sci-fi"))
        .temperature(0.9)  // High creativity
        .retries(2)
        .await?;

    println!("Character: {} (age {})", character.name, character.age);
    println!("Traits: {}", character.personality_traits.join(", "));
    println!("Backstory: {}\n", character.backstory);

    // Step 2: Write a scene
    let scene: Scene = agent
        .prompt(write_scene(&character, "discovering a hidden truth about their past"))
        .temperature(0.85)
        .await?;

    println!("📝 Scene");
    println!("Setting: {}", scene.setting);
    println!("\nDialogue:");
    for line in scene.dialogue {
        println!("  {}", line);
    }
    println!("\nAction: {}", scene.action);

    // Step 3: Use chat for revisions
    let mut chat = Agent::chat(std::env::var("GEMINI_API_KEY")?)
        .with_model("gemini-1.5-flash");

    let revision: String = chat
        .send(format!(
            "Here's a scene I wrote:\n\n{}\n\nHow can I make it more suspenseful?",
            scene.action
        ))
        .await?;

    println!("\n💡 Revision Suggestion:\n{}", revision);

    let improved: String = chat
        .send("Can you rewrite it with those improvements?")
        .await?;

    println!("\n✨ Improved Version:\n{}", improved);

    Ok(())
}
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

// Prompt execution
.prompt<T>(text: impl Into<String>) -> PromptBuilder<'_, T>
    where T: GeminiSchema + Deserialize + Send + 'static

// File operations
async .upload_file(data: &[u8], mime_type: impl Into<String>) -> Result<FileHandle>
async .delete_file(file: &FileHandle) -> Result<()>
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
```

### 5. Handle Errors Appropriately

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

## Examples

Check the `examples/` directory for more:

- `examples/ux.rs` - Comprehensive usage example
- Run with: `cargo run --example ux`

Named after the mythical giant **Adamastor** from Portuguese literature (_Os Lusíadas_ by Luís de Camões), who guards the Cape of Good Hope
