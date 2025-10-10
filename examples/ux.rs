use adamastor::{Agent, Result, schema};

#[schema]
struct Poem {
    title: String,
    content: String,
    /// The poetic style, e.g., "Haiku", "Limerick", "Free Verse"
    style: String,
}

#[schema]
struct BookIdea {
    title: String,
    logline: String,
    protagonist: String,
}

// A simple function to generate a prompt string
fn translate_poem(poem: &Poem, language: &str) -> String {
    format!(
        "Translate the following poem into {}:\n\n{}\n",
        language, poem.content
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("GEMINI_KEY").expect("GEMINI_API_KEY environment variable not set");

    // 1. AGENT SETUP: Configure the agent with global defaults.
    let agent = Agent::new(&api_key)
        .with_model("gemini-2.0-flash")
        .with_system_prompt("You are a helpful and creative assistant specializing in literature.")
        .with_requests_per_second(0.2);

    // 2. STRUCTURED OUTPUT: Get a Poem struct, overriding the temperature for this request.
    println!("--- Generating a Poem ---");
    let poem: Poem = agent
        .prompt("Write a short, epic poem about the Rust borrow checker.")
        .temperature(0.8) // More creative for this one task
        .await?;

    println!("Title: {}", poem.title);
    println!("Style: {}", poem.style);
    println!("{}\n", poem.content);

    // 3. TEXT OUTPUT (FROM FUNCTION): Use the generated poem to create a new prompt.
    println!("--- Translating the Poem ---");
    let japanese_translation: String = agent.prompt(translate_poem(&poem, "Japanese")).await?;

    println!("{}\n", japanese_translation);

    // 4. FILE HANDLING: Upload a file and use it in a prompt.
    // Note: Commented out since we need an actual file
    /*
    println!("--- Summarizing a Document ---");
    let file_data = std::fs::read("document.pdf")?;
    let file_handle = agent.upload_file(&file_data, "application/pdf").await?;

    let summary: String = agent
        .prompt("Summarize the key findings of the attached document.")
        .with_file(file_handle)
        .await?;

    println!("{}\n", summary);
    */

    // 5. CHAT: Maintain conversation history with a Chat object.
    println!("--- Starting a Chat Session ---");
    let mut chat = Agent::chat(&api_key);

    // First turn: structured output
    let idea: BookIdea = chat
        .send("Give me a novel sci-fi book idea about ancient AI.")
        .temperature(0.9)
        .await?;

    println!("Received Idea Title: {}", idea.title);
    println!("Logline: {}", idea.logline);
    println!("Protagonist: {}\n", idea.protagonist);

    // Second turn: text output with conversation context
    let follow_up: String = chat
        .send("That's a fascinating concept. Can you write the first paragraph of the book?")
        .await?;

    println!("Opening Paragraph:\n{}", follow_up);

    Ok(())
}
