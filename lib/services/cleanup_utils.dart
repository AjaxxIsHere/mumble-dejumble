/// Utilities shared by the LLM cleanup pipeline.
///
/// The system prompt below is part of the model contract: it is the exact
/// prompt the Qwen3.5 fine-tune was trained with (see
/// `Model Training Pipeline Final/dataset_creator.ipynb`). Changing it at
/// inference time degrades cleanup quality. Keep it byte-identical.
const String kCleanupSystemPrompt =
    'Clean up this voice dictation: remove filler words, stutters, and '
    'self-corrections; fix punctuation and capitalization. Keep every fact, '
    'name, date, time, and number exactly as dictated. Output only the '
    'cleaned text.';

/// Removes Qwen thinking blocks from generated output.
///
/// The model was trained with empty `<think></think>` prefixes, and at
/// inference it may still emit a (possibly non-empty) reasoning block before
/// the answer. Everything from the opening to the closing tag is discarded.
String stripThinkingBlocks(String text) {
  final cleaned = text.replaceAll(
    RegExp(r'<think>[\s\S]*?</think>', caseSensitive: false),
    '',
  );
  // Unterminated think block (generation hit the token cap mid-reasoning):
  // the visible answer would be incomplete, so drop everything after <think>.
  final openIndex = cleaned.toLowerCase().indexOf('<think>');
  if (openIndex >= 0) {
    return cleaned.substring(0, openIndex).trim();
  }
  return cleaned.trim();
}

/// Post-processes a raw LLM completion into the final polished text.
///
/// Returns an empty string when nothing usable remains, signalling the
/// caller to use its fallback path.
String postProcessCleanupOutput(String rawOutput) {
  var text = stripThinkingBlocks(rawOutput);
  // Some completions wrap the answer in quotes; dictation output should
  // never carry them.
  if (text.length >= 2 &&
      ((text.startsWith('"') && text.endsWith('"')) ||
          (text.startsWith('\u201C') && text.endsWith('\u201D')))) {
    text = text.substring(1, text.length - 1);
  }
  return text.trim();
}
