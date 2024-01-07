// This file has all the custom error types and other sturcts
// for working with the actual running for llm
// Also holds the function to make new threads to handle multiple requests
use std::{collections::HashSet, thread::JoinHandle};

use rand::SeedableRng;
use serenity::model::prelude::MessageId;
use thiserror::Error;

// Definition of custom error type InferenceError using the Error, Debug, and Clone traits
#[derive(Debug, Error, Clone)]
pub enum InferenceError {
    // Variant indicating that the generation process was cancelled
    #[error("The generation was cancelled.")]
    Cancelled,

    // Variant allowing for a custom error message with a placeholder ({0})
    #[error("{0}")]
    Custom(String),
}

// Implementation block for methods associated with InferenceError
impl InferenceError {
    // Constructor method for creating a custom InferenceError with a given message
    pub fn custom(s: impl Into<String>) -> Self {
        // Creating a new InferenceError with the Custom variant and the provided message
        Self::Custom(s.into())
    }
}

// struct representing a request for text generation
pub struct Request {
    // The input prompt for text generation
    pub prompt: String,
    // The size of the text generation batch
    pub batch_size: usize,
    // A channel sender for transmitting generated tokens
    // (In the realm of concurrent programming in Rust,
    // Flume channels provide a reliable means of communication
    // between different parts of a program running concurrently.
    // They facilitate the exchange of information among threads in a safe and organized manner)
    pub token_tx: flume::Sender<Token>,
    // The unique identifier for the associated Discord message
    pub message_id: MessageId,
    // An optional seed for the random number generator
    pub seed: Option<u64>,
}

// Definition of the Token enum, representing the result of text generation
pub enum Token {
    // Variant for a successfully generated token containing text
    Token(String),
    // Variant for an error during text generation, holding an InferenceError
    Error(InferenceError),
}

// This function is responsible for creating a new thread to handle text generation requests
pub fn make_thread(
    model: Box<dyn llm::Model>, // Takes a model implementing the llm::Model trait
    request_rx: flume::Receiver<Request>, // Receives requests through a channel
    cancel_rx: flume::Receiver<MessageId>, // Listens for cancellation signals associated with Discord messages
) -> JoinHandle<()> {
    // Spawns a new thread to continuously process incoming requests
    std::thread::spawn(move || loop {
        // Attempts to receive a text generation request from the channel
        if let Ok(request) = request_rx.try_recv() {
            // Processes the received request using the provided model
            match process_incoming_request(&request, model.as_ref(), &cancel_rx) {
                Ok(_) => {} // Do nothing if processing is successful
                Err(e) => {
                    // Sends an error token back through the communication channel if an error occurs
                    if let Err(err) = request.token_tx.send(Token::Error(e)) {
                        eprintln!("Failed to send error: {err:?}");
                    }
                }
            }
        }

        // Pauses the thread for a short duration to avoid excessive processing
        std::thread::sleep(std::time::Duration::from_millis(5));
    })
}

// Function to process incoming text generation requests
fn process_incoming_request(
    request: &Request,                      // Information about the generation request
    model: &dyn llm::Model,                 // The language model responsible for text generation
    cancel_rx: &flume::Receiver<MessageId>, // Channel for receiving cancellation signals
) -> Result<(), InferenceError> {
    // Creating a random number generator with an optional seed
    let mut rng = if let Some(seed) = request.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Starting a new session with the language model
    let mut session = model.start_session(Default::default());

    // Defining parameters for text generation
    let params = llm::InferenceParameters {
        sampler: llm::samplers::default_samplers(),
    };

    // Initiating the text generation process
    session
        .infer(
            model,
            &mut rng,
            &llm::InferenceRequest {
                // Converting the request prompt to the necessary format
                prompt: (&request.prompt).into(),
                parameters: &params,
                play_back_previous_tokens: false,
                maximum_token_count: None,
            },
            &mut Default::default(),
            // Callback function for handling each generated token
            move |t| {
                // Handling cancellation requests
                let cancellation_requests: HashSet<_> = cancel_rx.drain().collect();
                if cancellation_requests.contains(&request.message_id) {
                    // Signaling that the text generation is cancelled
                    return Err(InferenceError::Cancelled);
                }

                // Processing different types of generated tokens
                match t {
                    // For snapshot, prompt, and inferred tokens
                    llm::InferenceResponse::SnapshotToken(t)
                    | llm::InferenceResponse::PromptToken(t)
                    | llm::InferenceResponse::InferredToken(t) => {
                        // Sending the generated token through the channel
                        request
                            .token_tx
                            .send(Token::Token(t))
                            // Handling potential errors during token transmission
                            .map_err(|_| {
                                InferenceError::custom("Failed to send token to channel.")
                            })?;
                    }
                    // For end-of-text tokens
                    llm::InferenceResponse::EotToken => {}
                }

                // Indicating that the text generation process should continue
                Ok(llm::InferenceFeedback::Continue)
            },
        )
        // Ignoring the result, as only interested in potential errors
        .map(|_| ())
        // Converting specific types of errors into the custom InferenceError type for clarity
        .map_err(|e| match e {
            // If the error is due to a user callback
            llm::InferenceError::UserCallback(e) => {
                // Extracting and cloning the InferenceError from the user callback
                e.downcast::<InferenceError>().unwrap().as_ref().clone()
            }
            // For other types of errors
            e => InferenceError::custom(e.to_string()),
        })
}
