use clap::Parser;

fn main() {
    let cli = qwen_code::cli::Cli::parse();
    let code = qwen_code::cli::run(cli);
    std::process::exit(code);
}
