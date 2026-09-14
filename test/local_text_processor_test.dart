import 'package:flutter_test/flutter_test.dart';
import 'package:mumble_jumble/services/local_text_processor.dart';

void main() {
  group('LocalTextProcessor', () {
    final processor = LocalTextProcessor();

    setUp(() {
      processor.clearCache();
    });

    test('removes filler words and cleans punctuation', () {
      final raw = 'um hey can you send the report to Sarah tomorrow';
      final expected = 'Hey, can you send the report to Sarah tomorrow?';
      final actual = processor.normalizeForComparison(
        processor.cleanRawText(raw),
      );
      expect(actual, processor.normalizeForComparison(expected));
    });

    test('handles self correction', () {
      final raw = "I'll meet you at five actually six";
      final expected = "I'll meet you at six.";
      final actual = processor.normalizeForComparison(
        processor.cleanRawText(raw),
      );
      expect(actual, processor.normalizeForComparison(expected));
    });

    test('removes repetition and adds punctuation', () {
      final raw = 'uh I need milk bread eggs and coffee';
      final expected = 'I need milk, bread, eggs, and coffee.';
      final actual = processor.normalizeForComparison(
        processor.cleanRawText(raw),
      );
      expect(actual, processor.normalizeForComparison(expected));
    });

    test('preserves names and numbers', () {
      final raw = 'call John sorry call Sarah';
      final expected = 'Call Sarah.';
      final actual = processor.normalizeForComparison(
        processor.cleanRawText(raw),
      );
      expect(actual, processor.normalizeForComparison(expected));
    });

    test('keeps already clean text stable', () {
      final raw = 'Please send the summary by Friday.';
      final expected = 'Please send the summary by Friday.';
      final actual = processor.normalizeForComparison(
        processor.cleanRawText(raw),
      );
      expect(actual, processor.normalizeForComparison(expected));
    });
  });
}
