use adamastor::{schema, Agent, Result};

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

#[schema]
struct WeatherArgs {
    /// The city name, e.g., "Tokyo", "London", "San Francisco"
    location: String,
}

#[schema]
struct CalculatorArgs {
    /// The mathematical expression to evaluate, e.g., "2 + 2", "sqrt(16)"
    expression: String,
}

#[schema]
struct TravelPlan {
    destination: String,
    activities: Vec<String>,
    weather_info: String,
    packing_recommendations: Vec<String>,
}

// A simple function to generate a prompt string
fn translate_poem(poem: &Poem, language: &str) -> String {
    format!(
        "Translate the following poem into {}:\n\n{}\n",
        language, poem.content
    )
}

// Mock weather API call
async fn fetch_weather(location: &str) -> Result<String> {
    // In a real app, you would call an actual weather API here
    let weather_data = match location.to_lowercase().as_str() {
        "tokyo" => "Sunny with clear skies, 22°C. Perfect weather for sightseeing!",
        "london" => "Cloudy with occasional rain, 15°C. Bring an umbrella!",
        "san francisco" => "Partly cloudy, 18°C. Mild and pleasant.",
        "paris" => "Overcast with light drizzle, 16°C. Classic Parisian weather.",
        _ => "Weather data unavailable for this location. Conditions are moderate.",
    };

    Ok(format!("Weather in {}: {}", location, weather_data))
}

// Mock calculator
async fn calculate(expression: &str) -> Result<String> {
    // In a real app, you might use a proper expression parser
    let result = match expression {
        "2 + 2" => "4",
        "10 * 5" => "50",
        "100 / 4" => "25",
        _ => "Unable to calculate. Try a simpler expression.",
    };

    Ok(format!("Result: {}", result))
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("GEMINI_KEY").expect("GEMINI_KEY environment variable not set");

    println!("ADAMASTOR - Comprehensive Example");
    println!();

    // 1. AGENT SETUP: Configure the agent with global defaults.
    let agent = Agent::new(&api_key)
        .with_model("gemini-2.0-flash")
        .with_system_prompt(
            "You are a helpful and creative assistant specializing in literature and travel.",
        )
        .with_requests_per_second(0.2)
        .with_max_function_calls(10);

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

    // 4. TOOL CALLING: Use a tool to fetch real-time weather data
    println!("--- Using Tool Calling for Weather ---");
    let weather_response: String = agent
        .prompt("What's the weather like in Tokyo? Should I bring a jacket?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            fetch_weather(&args.location).await
        })
        .await?;

    println!("{}\n", weather_response);

    // 5. MULTIPLE TOOLS: Let the model choose which tool to use
    println!("--- Multiple Tools Available ---");
    let multi_tool_response: String = agent
        .prompt("What's 10 times 5, and what's the weather in London?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            fetch_weather(&args.location).await
        })
        .with_tool("calculator", |args: CalculatorArgs| async move {
            calculate(&args.expression).await
        })
        .with_max_function_calls(5)
        .await?;

    println!("{}\n", multi_tool_response);

    // 6. STRUCTURED OUTPUT WITH TOOLS: Get structured data that uses tool results
    println!("--- Structured Output Using Tools ---");
    let travel_plan: TravelPlan = agent
        .prompt("Create a weekend travel plan for Paris. Check the weather and suggest appropriate activities and packing items.")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            fetch_weather(&args.location).await
        })
        .temperature(0.7)
        .await?;

    println!("Destination: {}", travel_plan.destination);
    println!("Weather: {}", travel_plan.weather_info);
    println!("Activities:");
    for activity in &travel_plan.activities {
        println!("  - {}", activity);
    }
    println!("Packing Recommendations:");
    for item in &travel_plan.packing_recommendations {
        println!("  - {}", item);
    }
    println!();

    // 7. FILE HANDLING: Upload a file and use it in a prompt.
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

    // 8. CHAT: Maintain conversation history with a Chat object.
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

    println!("Opening Paragraph:\n{}\n", follow_up);

    // 9. CHAT WITH TOOLS: Tools work in stateful conversations too
    println!("--- Chat with Tool Calling ---");
    let mut weather_chat =
        Agent::chat(&api_key).with_system_prompt("You are a helpful travel assistant.");

    let first_response: String = weather_chat
        .send("I'm planning to visit San Francisco next week.")
        .await?;

    println!("Assistant: {}\n", first_response);

    let weather_advice: String = weather_chat
        .send("What's the weather like there? Should I pack warm clothes?")
        .with_tool("get_weather", |args: WeatherArgs| async move {
            fetch_weather(&args.location).await
        })
        .await?;

    println!("Assistant: {}\n", weather_advice);

    println!("All examples completed successfully!");

    Ok(())
}
