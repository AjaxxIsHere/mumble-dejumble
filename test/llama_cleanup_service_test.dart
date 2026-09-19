import 'package:flutter_test/flutter_test.dart';
import 'package:mumble_jumble/services/llama_cleanup_service.dart';
import 'package:mumble_jumble/services/local_text_processor.dart';

void main() {
  group('LlamaCleanupService (without native engine)', () {
    // The native llamadart backend is unavailable in unit tests, so the
    // service is always uninitialized here. This exercises the
    // graceful-degradation contract from plan.md §3: cleanup failure must
    // never lose user text.
    final service = LlamaCleanupService();
    final fallback = LocalTextProcessor();

    test('reports uninitialized and falls back to rule-based cleaning', () async {
      expect(service.initialized, isFalse);

      final out = await service.processText('um I I think we should go tomorrow');
      expect(out, isNotEmpty);
      expect(out, fallback.cleanRawText('um I I think we should go tomorrow'));
    });

    test('empty input returns empty output', () async {
      expect(await service.processText('   '), '');
    });
  });
}
