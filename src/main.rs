use async_openai::{
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use futures::StreamExt;
use std::{
    env,
    error::Error,
    io::{self, Write},
};
use tiktoken_rs::cl100k_base;

const PRICE_4: [f64; 2] = [3. / 1_000., 6. / 1_000.];
const PRICE_4T: [f64; 2] = [1. / 1_000., 3. / 1_000.];
const PRICE_3: [f64; 2] = [1. / 10_000., 2. / 10_000.];

#[derive(Clone, Debug)]
struct AppState {
    model: String,
    max_tokens: u16,
    auto_pipe: bool,
    context: Vec<ContextType>,
}

#[derive(Clone, Debug)]
enum ContextType {
    Assistant(String),
    User(String),
}

impl AppState {
    fn get_model(&self) -> String {
        self.model.clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    if env::var("OPENAI_API_KEY").is_err() {
        println!("The environment variable 'OPENAI_API_KEY' must be set to use this program.");
        return Ok(());
    }

    let client = Client::new();

    let mut state = AppState {
        model: String::from("gpt-3.5-turbo"),
        max_tokens: 512,
        auto_pipe: false,
        context: Vec::new(),
    };

    loop {
        print!("{}> ", state.get_model());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let mut input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }
        if input.starts_with(':') {
            parse_command(input, &mut state);
            continue;
        }
        // If the input doesn't start with a '|', then we can clear all context. Otherwise, we keep building the context.
        if !input.starts_with('|') && !state.auto_pipe {
            state.context = Vec::new();
        } else if input.starts_with('|') {
            input.remove(0);
        }
        state.context.push(ContextType::User(input));

        // Get input tokens
        let input_tokens = count_tokens_ctx(state.context.clone());

        let request = CreateChatCompletionRequestArgs::default()
            .model(state.get_model())
            .max_tokens(state.max_tokens)
            .messages(convert_context(state.context.clone()))
            .build()?;

        let mut stream = client.chat().create_stream(request).await?;

        let mut response_save = String::new();

        let mut lock = io::stdout().lock();
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    response.choices.iter().for_each(|chat_choice| {
                        if let Some(ref content) = chat_choice.delta.content {
                            write!(lock, "{}", content).unwrap();
                            response_save.push_str(content);
                        }
                    });
                }
                Err(err) => {
                    writeln!(lock, "An error occured: {err}").unwrap();
                }
            }
            io::stdout().flush()?;
        }
        println!();
        println!();
        let response_tokens = count_tokens(&response_save);
        state.context.push(ContextType::Assistant(response_save));

        let price = calc_price(input_tokens, response_tokens, &state);
        println!(
            "Prompt Tokens: {} | Completion Tokens: {} | Total Tokens: {} | Price: {:.5}p",
            input_tokens,
            response_tokens,
            input_tokens + response_tokens,
            price
        );
        println!();
    }
}

fn parse_command(command: String, state: &mut AppState) {
    let command = command.trim().to_lowercase();
    let (command, args) = {
        let mut words = command.split_whitespace();
        let cmd = words.next().unwrap();
        let args: String = words.collect();
        let args = args.trim().to_string();
        (cmd, args)
    };

    if command == ":q" || command == ":quit" {
        std::process::exit(0);
    }

    if command == ":c" || command == ":context" {
        state.auto_pipe = !state.auto_pipe;
        if state.auto_pipe {
            println!("Context will now be carried between messages.");
        } else {
            println!(
                "Context will no longer be carried between messages.\nPrefix messages with '|' to temporarily keep context."
            );
        }
        return;
    }

    if command == ":m" || command == ":model" {
        match args.as_str() {
            "3" => {
                state.model = "gpt-3.5-turbo".to_string();
                println!("Swapped to model gpt-3.5-turbo.");
            }
            "4" => {
                state.model = "gpt-4".to_string();
                println!("Swapped to model gpt-4.");
            }
            "4t" => {
                state.model = "gpt-4 turbo".to_string();
                println!("Swapped to model gpt-4 turbo.");
            }
            _ => println!("Unknown model. Please try again.\nPossible Options: 3, 4, 4t."),
        }
        return;
    }

    if command == ":h" || command == ":help" {
        println!("Commands:\n 1) :quit (q) - quits the program\n 2) :context (c) - toggles between keeping context and discarding it between messages.\n 3) :model (m) [3|4|4t]\n");
        return;
    }

    println!("Unknown command. Use :help to see a list of commands.");
    return;
}

fn convert_context(context: Vec<ContextType>) -> Vec<ChatCompletionRequestMessage> {
    let mut res = Vec::with_capacity(context.len());

    for ctx in context {
        let new = match ctx {
            ContextType::User(msg) => ChatCompletionRequestUserMessageArgs::default()
                .content(msg)
                .build()
                .unwrap()
                .into(),
            ContextType::Assistant(msg) => ChatCompletionRequestAssistantMessageArgs::default()
                .content(msg)
                .build()
                .unwrap()
                .into(),
        };

        res.push(new);
    }

    return res;
}

fn count_tokens_ctx(context: Vec<ContextType>) -> u16 {
    let mut sum = 0;
    for ctx in context {
        let msg = match ctx {
            ContextType::Assistant(m) => m,
            ContextType::User(m) => m,
        };
        sum += count_tokens(&msg);
    }

    sum
}

fn count_tokens(text: &str) -> u16 {
    let cl100k = cl100k_base().unwrap();
    let tokens = cl100k.split_by_token(&text, true).unwrap();
    return tokens.len() as u16;
}

fn calc_price(inp: u16, out: u16, state: &AppState) -> f64 {
    let inp = inp as f64;
    let out = out as f64;
    match state.model.as_str() {
        "gpt-3.5-turbo" => inp * PRICE_3[0] + out * PRICE_3[1],
        "gpt-4" => inp * PRICE_4[0] + out * PRICE_4[1],
        "gpt-4 turbo" => inp * PRICE_4T[0] + out * PRICE_4T[1],
        _ => 99999.0,
    }
}
