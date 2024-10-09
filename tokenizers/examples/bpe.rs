use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, TokenizerBuilder};

fn main() -> Result<()> {
    let vocab_size: usize = 50257;

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![AddedToken::from(String::from("<|endoftext|>"), true)])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![Strip::new(false, false).into()])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec!["wikitext.txt".to_string()])?
        .save("tokenizer.json", pretty)?;

    Ok(())
}
