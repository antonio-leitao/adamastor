<p align="center">
  <img src='assets/logo.svg' width='200px' align="center"></img>
</p>

<div align="center">
<h3 max-width='200px' align="center">Adamastor</h3>
  <p><i>Permutation Based Approximate Nearest Neighbors<br/>
  No index, infinitely incrementable<br/>
  Built with Rust</i><br/></p>
  <!-- <p> -->
<!-- <img alt="Pepy Total Downlods" src="https://img.shields.io/pepy/dt/caravela?style=for-the-badge&logo=python&labelColor=white&color=blue"> -->
  <!-- </p> -->
</div>

#

A Rust framework for building AI agents with the Gemini API.
Adamastor provides a type-safe, ergonomic interface for creating prompts, handling structured data, embeddings, and integrating tools with Gemini models.

## Features

- **Type-safe prompt chaining** - Output of one prompt becomes input to another
- **Structured schemas** - Define inputs and outputs with automatic JSON schema generation
- **Multiple named inputs** - Prompts and tools can accept multiple typed parameters
- **Tool integration** - Add custom tools that models can call during execution
- **Embeddings support** - Generate embeddings for documents and queries
- **File handling** - Upload and manage files for multimodal prompts
- **Rate limiting** - Built-in request rate management
- **Error handling** - Comprehensive error types for debugging

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
adamastor = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## Basic Usage

### Simple Agent Setup

```rust
use adamastor::{Agent, prompt, schema, Result};

#[schema]
struct Poem {
    title: String,
    content: String,
    style: String,
}

#[prompt]
fn write_poem() -> Poem {
    "Write a short poem about rust programming"
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .expect("GEMINI_API_KEY not found in environment.");

    let agent = Agent::new(&api_key);

    let poem = agent.prompt(write_poem).invoke(()).await?;
    println!("{}: {}", poem.title, poem.content);

    Ok(())
}
```

### Using Docstrings for Field Descriptions

The `#[schema]` macro automatically uses doc comments to provide descriptions to the Gemini model, improving its understanding of the desired structure.

```rust
#[schema]
struct CodeOutput {
    /// The generated source code, ready to be compiled or executed.
    code: String,

    /// A step-by-step explanation of how the code works.
    explanation: String,

    /// A rating of the code's complexity: "simple", "moderate", or "complex".
    complexity: String,
}
```

### Prompts with Input

When a prompt takes a single struct as input, simply pass that struct to the `invoke` method.

```rust
#[schema]
struct CodeRequest {
    /// The programming language to use (e.g., "Python", "Rust", "JavaScript")
    language: String,

    /// Detailed description of what the code should accomplish
    description: String,
}

#[prompt]
fn generate_code(req: CodeRequest) -> CodeOutput {
    format!(
        "Generate {} code that {}.\n\
         Include an explanation and rate the complexity.",
        req.language, req.description
    )
}

// Usage
let request = CodeRequest {
    language: "Python".to_string(),
    description: "sorts a list of numbers".to_string(),
};

let result = agent.prompt(generate_code).invoke(request).await?;
println!("--- Code ---\n{}\n\n--- Explanation ---\n{}", result.code, result.explanation);
```

### Multiple Named Inputs for Complex Flows

When a prompt requires multiple distinct inputs, simply pass them as a tuple to the `invoke` method. Adamastor handles the bundling behind the scenes.

```rust
use std::collections::HashMap;

#[schema]
struct Document {
    content: String,
    metadata: HashMap<String, String>,
}

#[schema]
struct Analysis {
    comparison: String,
    similarities: Vec<String>,
    differences: Vec<String>,
}

// This prompt takes two Documents and a focus String
#[prompt]
fn compare_documents(doc1: Document, doc2: Document, focus: String) -> Analysis {
    format!(
        "Compare these documents focusing on {}:\n\n\
         Doc1: {}\n\n\
         Doc2: {}\n",
        focus, doc1.content, doc2.content
    )
}

// Usage
let doc1 = Document { content: "Rust uses a borrow checker for memory safety.".to_string(), ..Default::default() };
let doc2 = Document { content: "Go uses a garbage collector for memory management.".to_string(), ..Default::default() };
let focus = "memory management approach".to_string();

// Pass the arguments in a tuple
let analysis = agent
    .prompt(compare_documents)
    .invoke((doc1, doc2, focus))
    .await?;

println!("Comparison: {}", analysis.comparison);
```

### Prompt Chaining - The Core Pattern

The real power of Adamastor is type-safe prompt chaining. The output struct of one prompt can be the input of another, creating complex, multi-step agents that are easy to reason about.

```rust
#[schema]
struct Article {
    title: String,
    content: String,
}

#[schema]
struct Summary {
    one_line: String,
    main_topics: Vec<String>,
}

#[schema]
struct StudyGuide {
    topic: String,
    questions: Vec<String>,
    key_concepts: Vec<String>,
}

// First prompt: Generate an article from a topic
#[prompt]
fn write_article(topic: String) -> Article {
    format!("Write a detailed article about {}", topic)
}

// Second prompt: Takes the entire Article struct as input, outputs a Summary
#[prompt]
fn summarize(article: Article) -> Summary {
    format!(
        "Summarize this article:\nTitle: {}\nContent: {}",
        article.title, article.content
    )
}

// Third prompt: Takes the Summary as input, outputs a StudyGuide
#[prompt]
fn create_study_guide(summary: Summary) -> StudyGuide {
    format!(
        "Create a study guide based on this summary:\nOne-liner: {}\nTopics: {:?}",
        summary.one_line, summary.main_topics
    )
}

// Chain them together with compile-time type safety
let article = agent
    .prompt(write_article)
    .invoke("Rust's ownership model".to_string())
    .await?;

let summary = agent
    .prompt(summarize)
    .invoke(article)  // The `Article` output flows directly into the next prompt
    .await?;

let study_guide = agent
    .prompt(create_study_guide)
    .invoke(summary)  // The `Summary` output flows into the final prompt
    .await?;

println!("Study Guide Questions: {:#?}", study_guide.questions);
```

### Primitive Return Types

You can use Rust primitive types directly as inputs and outputs without needing the `#[schema]` macro.

```rust
#[prompt]
fn calculate_sum(numbers: Vec<i32>) -> f64 {
    format!("Calculate the sum of these numbers: {:?}", numbers)
}

#[prompt]
fn get_word_count(text: String) -> u32 {
    format!("Count the words in this text: {}", text)
}

#[prompt]
fn check_validity(data: String) -> bool {
    format!("Is this valid JSON: {}", data)
}
```

## Embeddings

Adamastor provides powerful embedding capabilities for semantic search, similarity comparisons, and RAG applications.

```rust
// Single document embedding (default: for retrieval storage)
let embedding: Vec<f32> = agent
    .embed("This is my document content.")
    .invoke()
    .await?;

// Batch embeddings for multiple documents
let embeddings: Vec<Vec<f32>> = agent
    .embed_batch(&["First document", "Second document", "Third document"])
    .invoke()
    .await?;

// Query embedding (optimized for search queries)
let query_embedding: Vec<f32> = agent
    .embed("What is the meaning of life?")
    .as_query()
    .invoke()
    .await?;

// Reduced dimensionality (saves storage and bandwidth)
let compact_embedding: Vec<f32> = agent
    .embed("Large text to embed...")
    .with_dim(768)  // Options: 768, 1536, or 3072 (default)
    .invoke()
    .await?;
```

### Embedding Example: Semantic Search

```rust
use adamastor::{Agent, Result};

async fn semantic_search(
    agent: &Agent,
    documents: &[String],
    query: &str,
) -> Result<Vec<(usize, f32)>> {
    let doc_embeddings = agent.embed_batch(documents).invoke().await?;
    let query_embedding = agent.embed(query).as_query().invoke().await?;

    let mut similarities: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(idx, doc_emb)| (idx, cosine_similarity(&query_embedding, doc_emb)))
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(similarities)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```

## Advanced Usage with Tools

### Defining Tools

Tools are async functions that the model can call to interact with external systems. Use the `#[tool]` macro and provide clear doc comments.

```rust
use std::error::Error;

#[schema]
struct WebQuery {
    /// The specific search query string.
    query: String,
    /// Maximum number of search results to return.
    max_results: u32,
}

#[tool]
async fn search_web(input: WebQuery) -> Result<String, Box<dyn Error + Send + Sync>> {
    // In a real application, you would call a search engine API here.
    Ok(format!("Found {} results for: '{}'", input.max_results, input.query))
}
```

### Tools with Multiple Parameters

Just like with prompts, if a tool has multiple arguments, they will be bundled automatically. The calling convention from the agent's perspective remains the same.

```rust
#[tool]
async fn database_query(
    table: String,
    columns: Vec<String>,
    limit: u32,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    // In a real application, you would connect to a database here.
    Ok(format!(
        "SELECT {} FROM {} LIMIT {}",
        columns.join(", "),
        table,
        limit
    ))
}
```

### Using Tools in Prompts

You can add tools to a specific prompt execution or configure an agent with persistent tools that are always available.

```rust
// Add tools to a specific prompt
let result = agent
    .prompt(solve_problem)
    .with_tool(calculator)
    .with_tool(search_web)
    .invoke(input)
    .await?;

// Or, create an agent with persistent tools
let agent = Agent::new(&api_key)
    .with_tool(calculator)
    .with_tool(search_web);

// Now all prompts on this agent can use these tools by default.
```

## Configuration Reference

### Agent Configuration

```rust
let agent = Agent::new(&api_key)
    // Model to use, e.g., "gemini-1.5-flash", "gemini-1.5-pro".
    .with_model("gemini-2.5-flash")
    // A system prompt sets a consistent role or behavior.
    .with_system_prompt("You are a helpful coding assistant.")
    // Adjust requests per second to stay within API limits (default: 2.0).
    .with_requests_per_second(3.0)
    // Prevent infinite loops in tool usage (default: 10).
    .with_max_function_calls(15)
    // Add persistent tools available to all prompts.
    .with_tool(calculator);
```

### Prompt Configuration

Fine-tune individual prompt executions by chaining methods.

```rust
let result = agent
    .prompt(my_prompt)
    // Controls randomness (0.0 = deterministic, 1.0 = creative).
    .temperature(0.8)
    // Limits response length to control costs and size.
    .max_tokens(500)
    // An alternative to temperature for sampling.
    .top_p(0.95)
    // Number of retries on transient failures (default: 1, no retries).
    .retries(3)
    // Override the agent's default max function calls.
    .max_function_calls(20)
    // Add tools specific to this prompt run.
    .with_tool(special_tool)
    // Attach files for multimodal input.
    .with_file(file_handle)
    .invoke(input)
    .await?;
```

## File Handling

Upload files to use with multimodal prompts.

```rust
// Upload file contents and get a handle
let file_data = std::fs::read("my_document.txt")?;
let file_handle = agent
    .upload_file(&file_data, "text/plain")
    .await?;

// Use the handle in a prompt
let result = agent
    .prompt(analyze_document)
    .with_file(file_handle.clone())
    .invoke("Summarize this file.".to_string())
    .await?;

// Clean up the file from the server
agent.delete_file(&file_handle).await?;
```

## Error Handling

Adamastor provides specific error types for robust error handling.

```rust
use adamastor::{AdamastorError, Result};

match agent.prompt(my_prompt).invoke(input).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(AdamastorError::ToolNotFound(name)) => {
        eprintln!("Error: Tool '{}' was called but not found.", name)
    },
    Err(AdamastorError::MaxFunctionCalls(max)) => {
        eprintln!("Error: Exceeded the limit of {} function calls.", max)
    },
    Err(AdamastorError::RateLimit) => {
        eprintln!("Error: Rate limit exceeded. Try again later.")
    },
    Err(AdamastorError::Api(msg)) => {
        eprintln!("Error: An API error occurred: {}", msg)
    },
    Err(e) => eprintln!("An unexpected error occurred: {}", e),
}
```
