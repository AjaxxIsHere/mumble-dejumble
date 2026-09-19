import 'package:flutter_test/flutter_test.dart';
import 'package:mumble_jumble/services/cleanup_utils.dart';

void main() {
  group('kCleanupSystemPrompt', () {
    test('matches the training-time prompt byte-for-byte', () {
      // This exact string is part of the model contract (see
      // `Model Training Pipeline Final/dataset_creator.ipynb`, SYSTEM_PROMPT
      // as rendered in the chat-template sample). If this test fails,
      // inference-time cleanup quality will silently degrade.
      const expected =
          'Clean up this voice dictation: remove filler words, stutters, and '
          'self-corrections; fix punctuation and capitalization. Keep every '
          'fact, name, date, time, and number exactly as dictated. Output '
          'only the cleaned text.';
      expect(kCleanupSystemPrompt, expected);
    });
  });

  group('stripThinkingBlocks', () {
    test('removes an empty think block', () {
      final out = stripThinkingBlocks('<think>\n\n</think>\nHello world.');
      expect(out, 'Hello world.');
    });

    test('removes a non-empty think block', () {
      final out = stripThinkingBlocks(
        '<think>The user said "um" twice.</think>Call Sarah.',
      );
      expect(out, 'Call Sarah.');
    });

    test('handles multi-line reasoning blocks', () {
      final out = stripThinkingBlocks(
        '<think>\nstep one\nstep two\n</think>\nWe must go tomorrow.',
      );
      expect(out, 'We must go tomorrow.');
    });

    test('drops unterminated think blocks entirely', () {
      // Generation hit the token cap mid-reasoning: the visible answer would
      // be incomplete, so everything from <think> onwards is discarded.
      final out = stripThinkingBlocks('Clean text so far <think>but then');
      expect(out, 'Clean text so far');
    });

    test('leaves plain output untouched', () {
      expect(stripThinkingBlocks('Plain answer.'), 'Plain answer.');
    });
  });

  group('postProcessCleanupOutput', () {
    test('unwraps straight-quoted output', () {
      expect(postProcessCleanupOutput('"We must go tomorrow."'),
          'We must go tomorrow.');
    });

    test('unwraps curly-quoted output', () {
      expect(postProcessCleanupOutput('\u201CWe must go tomorrow.\u201D'),
          'We must go tomorrow.');
    });

    test('returns empty string when only a think block remains', () {
      expect(postProcessCleanupOutput('<think>reasoning only</think>'), '');
    });

    test('passes through normal completions', () {
      expect(
        postProcessCleanupOutput(
            '<think></think>Hey Sarah, can you reschedule to Friday at 3:30 PM?'),
        'Hey Sarah, can you reschedule to Friday at 3:30 PM?',
      );
    });
  });
}
