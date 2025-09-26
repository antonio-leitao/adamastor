<p align="center">
  <img src='assets/logo.svg' width='200px' align="center"></img>
</p>

<div align="center">
<h3 max-width='200px' align="center">Adamastor</h3>
  <p><i>Rust framework for building LLM Agents<br/>
  Type-safe for creating structured adn reliable prompts<br/>
  Built with Rust</i><br/></p>
  <!-- <p> -->
<!-- <img alt="Pepy Total Downlods" src="https://img.shields.io/pepy/dt/caravela?style=for-the-badge&logo=python&labelColor=white&color=blue"> -->
  <!-- </p> -->
</div>

<div align="right">
    <i>«Eu sou aquele oculto e grande Cabo<br>
A quem chamais vós outros Tormentório,<br>
Que nunca a Ptolomeu, Pompónio, Estrabo,<br>
Plínio e quantos passaram fui notório.<br>
Aqui toda a Africana costa acabo<br>
Neste meu nunca visto Promontório,<br>
Que pera o Pólo Antártico se estende,<br>
A quem vossa ousadia tanto ofende.<br>
    </i></div>

## Features

- **Flexible prompt system** - Use typed prompts, strings, or closures
- **Type-safe prompt chaining** - Output of one prompt becomes input to another
- **Structured schemas** - Define inputs and outputs with automatic JSON schema generation
- **Schema override** - Change expected output types at runtime
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
    let api_key = std::env::var("GEMINI_API_KEY")?;
    let agent = Agent::new(&api_key);

    let poem = agent.prompt(write_poem).invoke(()).await?;
    println!("{}: {}", poem.title, poem.content);

    Ok(())
}
```

## Flexible Prompt System

### 1. String Prompts (Ad-hoc)

Use simple strings when you need quick, one-off prompts:

```rust
// Type inference from variable annotation
let recipe: Recipe = agent
    .prompt("Create a recipe for chocolate chip cookies")
    .invoke()
    .await?;

// Turbofish syntax
let haiku = agent
    .prompt("Write a haiku about the ocean")
    .invoke::<Haiku>()
    .await?;

// Explicit schema with .returns()
let doc = agent
    .prompt("Write technical documentation about REST APIs")
    .returns::<TechnicalDoc>()
    .temperature(0.7)
    .invoke(())
    .await?;
```

### 2. On-the-fly Prompts

Generate prompts on-the-fly with strings that capture context:

```rust
let cuisine = "Italian";
let recipe = agent
    .prompt(format!("Create a traditional {} recipe", cuisine))
    .invoke::<Recipe>()
    .await?;

// Capturing multiple variables
let ingredient = "tomatoes";
let season = "summer";
let dish = agent
    .prompt(
        format!("Create a {} recipe featuring {} as the main ingredient",
            season, ingredient)
    )
    .invoke::<SimpleRecipe>()
    .await?;
```

### 3. Typed Prompts (Best for Chaining)

Define reusable, type-safe prompts with the `#[prompt]` macro:

```rust
#[prompt]
fn generate_code(req: CodeRequest) -> CodeOutput {
    format!("Generate {} code that {}", req.language, req.description)
}

let result = agent
    .prompt(generate_code)
    .invoke(CodeRequest {
        language: "Python".to_string(),
        description: "sorts a list".to_string(),
    })
    .await?;
```

## Schema Override

Change the expected output type of any prompt:

```rust
// Override a typed prompt's output
let simple_version = agent
    .prompt(write_article)           // Normally returns Article
    .returns::<SimpleRecipe>()       // Override to return SimpleRecipe
    .invoke("pasta carbonara".to_string())
    .await?;

// Make a string prompt chainable by specifying its type
let typed_for_chain = agent
    .prompt("Extract key points about async programming")
    .returns::<Article>()             // Now it can be chained!
    .temperature(0.5)
    .invoke(())
    .await?;
```

## Prompt Chaining

### Typed Prompt Chaining

Chain typed prompts for complex, multi-step operations:

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

#[prompt]
fn write_article(topic: String) -> Article {
    format!("Write a detailed article about {}", topic)
}

#[prompt]
fn summarize(article: Article) -> Summary {
    format!("Summarize this article: {}", article.title)
}

#[prompt]
fn create_study_guide(summary: Summary) -> StudyGuide {
    format!("Create a study guide from: {}", summary.one_line)
}

// Chain with .then() method
let guide = agent
    .prompt(write_article)
    .then(summarize)
    .then(create_study_guide)
    .invoke("Design patterns in Rust".to_string())
    .await?;
```

### Mixed Chaining

Start with strings, add types, then chain:

```rust
// String → Typed → Typed
let article = agent
    .prompt("Write about the benefits of static typing")
    .returns::<Article>()        // Specify type for string prompt
    .invoke(())
    .await?;

let summary = agent
    .prompt(summarize)           // Use typed prompt
    .invoke(article)
    .await?;

let guide = agent
    .prompt(create_study_guide)  // Chain another typed prompt
    .invoke(summary)
    .await?;
```

## Multiple Named Inputs

Prompts can accept multiple distinct inputs:

```rust
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

#[prompt]
fn compare_documents(doc1: Document, doc2: Document, focus: String) -> Analysis {
    format!("Compare these documents focusing on {}", focus)
}

// Pass multiple arguments as a tuple
let analysis = agent
    .prompt(compare_documents)
    .invoke((doc1, doc2, "writing style".to_string()))
    .await?;
```

## Primitive Types

Use Rust primitive types directly without needing the `#[schema]` macro:

```rust
#[prompt]
fn calculate_sum(numbers: Vec<i32>) -> f64 {
    format!("Calculate the sum of: {:?}", numbers)
}

#[prompt]
fn get_word_count(text: String) -> u32 {
    format!("Count words in: {}", text)
}

#[prompt]
fn is_valid_json(data: String) -> bool {
    format!("Is this valid JSON: {}", data)
}

// Usage
let sum = agent.prompt(calculate_sum).invoke(vec![1, 2, 3]).await?;
let count = agent.prompt(get_word_count).invoke("Hello world".to_string()).await?;
let valid = agent.prompt(is_valid_json).invoke("{}".to_string()).await?;
```

## Using Docstrings

Doc comments are automatically used as field descriptions:

```rust
#[schema]
struct CodeOutput {
    /// The generated source code, ready to compile
    code: String,

    /// Step-by-step explanation of the code
    explanation: String,

    /// Complexity rating: "simple", "moderate", or "complex"
    complexity: String,
}
```

## Embeddings

Generate embeddings for semantic search and RAG:

```rust
// Single document embedding
let embedding = agent
    .embed("Document content")
    .invoke()
    .await?;

// Batch embeddings
let embeddings = agent
    .embed_batch(&["Doc 1", "Doc 2", "Doc 3"])
    .invoke()
    .await?;

// Query embedding (optimized for search)
let query_embedding = agent
    .embed("What is the meaning of life?")
    .as_query()
    .invoke()
    .await?;

// Reduced dimensionality
let compact = agent
    .embed("Large text...")
    .with_dim(768)  // 768, 1536, or 3072 (default)
    .invoke()
    .await?;
```

## Tools

### Defining Tools

```rust
#[schema]
struct WebQuery {
    /// The search query string
    query: String,
    /// Maximum results to return
    max_results: u32,
}

#[tool]
async fn search_web(input: WebQuery) -> Result<String, Box<dyn Error + Send + Sync>> {
    Ok(format!("Found {} results for '{}'", input.max_results, input.query))
}

// Tools with multiple parameters
#[tool]
async fn database_query(
    table: String,
    columns: Vec<String>,
    limit: u32,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    Ok(format!("SELECT {} FROM {} LIMIT {}",
        columns.join(", "), table, limit))
}
```

### Using Tools

```rust
// Add tools to specific prompts
let result = agent
    .prompt(solve_problem)
    .with_tool(search_web)
    .with_tool(calculator)
    .invoke(input)
    .await?;

// Or create an agent with persistent tools
let agent = Agent::new(&api_key)
    .with_tool(calculator)
    .with_tool(search_web);
```

## Configuration

### Agent Configuration

```rust
let agent = Agent::new(&api_key)
    .with_model("gemini-2.0-flash")
    .with_system_prompt("You are a helpful assistant")
    .with_requests_per_second(3.0)
    .with_max_function_calls(15)
    .with_tool(calculator);
```

### Prompt Configuration

```rust
let result = agent
    .prompt(my_prompt)
    .temperature(0.8)           // Randomness (0.0-1.0)
    .max_tokens(500)            // Response length limit
    .top_p(0.95)               // Alternative to temperature
    .retries(3)                // Retry on failure
    .max_function_calls(20)    // Tool call limit
    .with_tool(special_tool)   // Add tool for this prompt
    .with_file(file_handle)    // Attach files
    .invoke(input)
    .await?;
```

## File Handling

```rust
// Upload file
let file_data = std::fs::read("document.txt")?;
let file_handle = agent
    .upload_file(&file_data, "text/plain")
    .await?;

// Use in prompt
let result = agent
    .prompt(analyze_document)
    .with_file(file_handle.clone())
    .invoke("Summarize this file")
    .await?;

// Clean up
agent.delete_file(&file_handle).await?;
```

## Error Handling

```rust
use adamastor::{AdamastorError, Result};

match agent.prompt(my_prompt).invoke(input).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(AdamastorError::ToolNotFound(name)) => {
        eprintln!("Tool '{}' not found", name)
    },
    Err(AdamastorError::MaxFunctionCalls(max)) => {
        eprintln!("Exceeded {} function calls", max)
    },
    Err(AdamastorError::RateLimit) => {
        eprintln!("Rate limit exceeded")
    },
    Err(e) => eprintln!("Error: {}", e),
}
```

## Complete Example

```rust
use adamastor::{Agent, prompt, schema, Result};
use std::collections::HashMap;

#[schema]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new(std::env::var("GEMINI_API_KEY")?)
        .with_model("gemini-2.0-flash");

    // 1. Simple string prompt
    let recipe: Recipe = agent
        .prompt("Create a healthy breakfast recipe")
        .temperature(0.7)
        .invoke()
        .await?;

    println!("Recipe: {}", recipe.name);

    // 2. Dynamic prompt with closure
    let meal_type = "dinner";
    let recipe = agent
        .prompt(|| format!("Create a quick {} recipe", meal_type))
        .invoke::<Recipe>()
        .await?;

    // 3. Typed prompt with chaining
    #[prompt]
    fn improve_recipe(recipe: Recipe) -> Recipe {
        format!("Make this recipe healthier: {}", recipe.name)
    }

    let improved = agent
        .prompt(improve_recipe)
        .invoke(recipe)
        .await?;

    println!("Improved: {}", improved.name);

    Ok(())
}
```

## License

MIT
