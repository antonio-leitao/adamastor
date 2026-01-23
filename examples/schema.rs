//! Schema Demo - Demonstrates robust Gemini schema generation
//!
//! Run with: cargo run --example schema_demo

use adamastor::GeminiSchema;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ============ Example 1: Simple Struct ============

/// A movie review with rating
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MovieReview {
    /// The title of the movie being reviewed
    title: String,
    /// Release year
    year: i32,
    /// Rating from 1.0 to 10.0
    rating: f32,
    /// Brief summary of the review
    summary: String,
    /// Whether you'd recommend this movie
    recommended: bool,
}

// ============ Example 2: With Optional Fields ============

/// A person's profile with some optional information
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct PersonProfile {
    /// Full name (required)
    name: String,
    /// Age in years (optional - may not be known)
    age: Option<i32>,
    /// Current occupation (optional - might be unemployed/retired)
    occupation: Option<String>,
    /// Short biography
    bio: String,
    /// List of hobbies
    hobbies: Vec<String>,
}

// ============ Example 3: Nested Structs ============

/// An ingredient in a recipe
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Ingredient {
    /// Name of the ingredient
    name: String,
    /// Quantity with units (e.g., "2 cups", "100g")
    quantity: String,
    /// Whether this ingredient is optional
    optional: bool,
}

/// A complete recipe
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Recipe {
    /// Name of the dish
    name: String,
    /// Brief description
    description: String,
    /// List of ingredients needed
    ingredients: Vec<Ingredient>,
    /// Step-by-step cooking instructions
    instructions: Vec<String>,
    /// Preparation time in minutes (optional)
    prep_time_minutes: Option<i32>,
    /// Cooking time in minutes (optional)
    cook_time_minutes: Option<i32>,
    /// Number of servings
    servings: i32,
}

// ============ Example 4: Enums ============

/// Difficulty level for a task
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// A coding challenge
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CodingChallenge {
    /// Title of the challenge
    title: String,
    /// Problem description
    description: String,
    /// Difficulty level
    difficulty: Difficulty,
    /// Expected time to solve in minutes
    time_estimate_minutes: i32,
    /// Topics/tags for the challenge
    topics: Vec<String>,
}

// ============ Example 5: Sentiment Analysis ============

/// Sentiment analysis result
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SentimentAnalysis {
    /// The analyzed text
    text: String,
    /// Overall sentiment: positive, negative, or neutral
    sentiment: String,
    /// Confidence score from 0.0 to 1.0
    confidence: f32,
    /// Key phrases that influenced the sentiment
    key_phrases: Vec<String>,
    /// Detected emotions (optional)
    emotions: Option<Vec<String>>,
}

fn test() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Adamastor Schema Demo - Robust Generation          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ============ MovieReview Schema ============
    println!("1. MovieReview Schema (Simple struct)");
    println!("   ─────────────────────────────────");
    let schema = MovieReview::gemini_schema();
    println!("{}\n", serde_json::to_string_pretty(&schema).unwrap());

    // ============ PersonProfile Schema ============
    println!("2. PersonProfile Schema (With Optional fields)");
    println!("   ──────────────────────────────────────────");
    let schema = PersonProfile::gemini_schema();
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    // Verify optional fields handling
    let required = schema.get("required").unwrap().as_array().unwrap();
    println!("\n   ✓ Required fields: {:?}", required);
    println!("   ✓ Note: 'age' and 'occupation' are NOT in required (they're Option<T>)");

    let props = schema.get("properties").unwrap();
    let age_nullable = props.get("age").unwrap().get("nullable");
    println!("   ✓ 'age' has nullable: {:?}\n", age_nullable);

    // ============ Recipe Schema ============
    println!("3. Recipe Schema (Nested structs with Vec)");
    println!("   ─────────────────────────────────────────");
    let schema = Recipe::gemini_schema();
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    // Verify nested propertyOrdering
    let props = schema.get("properties").unwrap();
    let ingredients = props.get("ingredients").unwrap();
    let items = ingredients.get("items").unwrap();
    let nested_ordering = items.get("propertyOrdering");
    println!(
        "\n   ✓ Nested Ingredient struct has propertyOrdering: {:?}\n",
        nested_ordering
    );

    // ============ CodingChallenge Schema ============
    println!("4. CodingChallenge Schema (With enum)");
    println!("   ──────────────────────────────────");
    let schema = CodingChallenge::gemini_schema();
    println!("{}\n", serde_json::to_string_pretty(&schema).unwrap());

    // ============ SentimentAnalysis Schema ============
    println!("5. SentimentAnalysis Schema (Mixed types)");
    println!("   ──────────────────────────────────────");
    let schema = SentimentAnalysis::gemini_schema();
    println!("{}\n", serde_json::to_string_pretty(&schema).unwrap());

    // ============ Summary ============
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Robustness Features                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ ✓ Optional<T> fields NOT in 'required' array                 ║");
    println!("║ ✓ Optional<T> fields have 'nullable: true'                   ║");
    println!("║ ✓ All objects have 'propertyOrdering' for consistent output  ║");
    println!("║ ✓ Nested structs also get propertyOrdering                   ║");
    println!("║ ✓ No unsupported fields ($schema, $id, definitions, etc.)    ║");
    println!("║ ✓ Doc comments become field descriptions                     ║");
    println!("║ ✓ Enums properly serialized                                  ║");
    println!("║ ✓ $ref references resolved and inlined                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

use adamastor::Agent;

#[tokio::main]
async fn main() -> adamastor::Result<()> {
    test();
    let api_key = std::env::var("GEMINI_KEY").expect("Set GEMINI_API_KEY environment variable");

    let agent = Agent::new(api_key).with_model("gemini-2.5-flash");

    // Structured output - type-safe!
    let review: MovieReview = agent
        .prompt("Review the movie 'Inception' by Christopher Nolan")
        .temperature(0.3)
        .await?;

    println!("Title: {}", review.title);
    println!("Year: {}", review.year);
    println!("Rating: {}/10", review.rating);
    println!("Summary: {}", review.summary);
    println!("Recommended: {}", review.recommended);

    // With optional fields
    let profile: PersonProfile = agent
        .prompt("Create a profile for a fictional software engineer named Alice")
        .await?;

    println!("\nName: {}", profile.name);
    println!("Age: {:?}", profile.age); // Option<i32>
    println!("Occupation: {:?}", profile.occupation); // Option<String>
    println!("Bio: {}", profile.bio);
    println!("Hobbies: {:?}", profile.hobbies);

    // Nested structs
    let recipe: Recipe = agent
        .prompt("Give me a recipe for chocolate chip cookies")
        .await?;

    println!("\nRecipe: {}", recipe.name);
    println!("Servings: {}", recipe.servings);
    for (i, ingredient) in recipe.ingredients.iter().enumerate() {
        println!("  {}. {} - {}", i + 1, ingredient.name, ingredient.quantity);
    }

    Ok(())
}
