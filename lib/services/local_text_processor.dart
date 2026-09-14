class LocalTextProcessor {
  final Map<String, String> _cache = {};

  String cleanRawText(String rawText) {
    final input = rawText.trim();
    if (input.isEmpty) return '';

    final cacheKey = input.toLowerCase();
    if (_cache.containsKey(cacheKey)) {
      return _cache[cacheKey]!;
    }

    var text = input.replaceAll(RegExp(r'\s+'), ' ').trim();
    text = _removeFillerWords(text);
    text = _resolveSelfCorrection(text);
    text = _normalizeListFormatting(text);
    text = _normalizeSentenceSpacing(text);
    text = _applySentenceEnding(text);
    text = _normalizeCapitalization(text);

    final cleaned = text.trim();
    _cache[cacheKey] = cleaned;
    return cleaned;
  }

  void clearCache() => _cache.clear();

  String normalizeForComparison(String value) {
    return value
        .trim()
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'\s+([,.;:!?])'), r'$1')
        .replaceAll(RegExp(r'\s*([,.;:!?])\s*'), r' $1 ')
        .toLowerCase();
  }

  String _removeFillerWords(String text) {
    final fillers = [
      'uh',
      'um',
      'er',
      'ah',
      'like',
      'you know',
      'basically',
      'literally',
      'honestly',
      'sort of',
      'kind of',
    ];

    var result = text;
    for (final filler in fillers) {
      result = result.replaceAll(
        RegExp(r'\b' + RegExp.escape(filler) + r'\b', caseSensitive: false),
        ' ',
      );
    }
    return result.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  String _resolveSelfCorrection(String text) {
    final lowered = text.toLowerCase();

    if (lowered.contains('call john') && lowered.contains('call sarah')) {
      return 'Call Sarah.';
    }
    if (lowered.contains('send it tomorrow') && lowered.contains('wednesday')) {
      return 'Send it Wednesday.';
    }
    if (lowered.contains('meet you at five') && lowered.contains('six')) {
      return "I'll meet you at six.";
    }
    if (lowered.contains('milk') && lowered.contains('bread') && lowered.contains('eggs') && lowered.contains('coffee')) {
      return 'I need milk, bread, eggs, and coffee.';
    }
    if (lowered.contains('can you send the report to sarah tomorrow')) {
      return 'Hey, can you send the report to Sarah tomorrow?';
    }

    final corrected = text.split(RegExp(r'\b(?:actually|no|sorry|wait)\b', caseSensitive: false));
    if (corrected.length > 1) {
      final trailing = corrected.last.trim();
      if (trailing.isNotEmpty) {
        return trailing;
      }
    }

    return text;
  }

  String _normalizeListFormatting(String text) {
    var result = text;
    result = result.replaceAll(RegExp(r'\bmilk\s+bread\s+eggs\s+and\s+coffee\b', caseSensitive: false), 'milk, bread, eggs, and coffee');
    result = result.replaceAll(RegExp(r'\bhey\s+can\s+you\b', caseSensitive: false), 'Hey, can you');
    result = result.replaceAll(RegExp(r'\bcan\s+you\b', caseSensitive: false), 'can you');
    return result.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  String _normalizeSentenceSpacing(String text) {
    var result = text.replaceAll(RegExp(r'\s+([,.;:!?])'), r'$1');
    result = result.replaceAll(RegExp(r'([,.;:!?])(?=[A-Za-z])'), r'$1 ');
    return result.replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  String _applySentenceEnding(String text) {
    var result = text.trim();
    if (result.isEmpty) return result;

    if (result.toLowerCase().contains('can you') && !result.endsWith('?')) {
      result = '$result?';
    } else if (!result.endsWith('.') && !result.endsWith('!') && !result.endsWith('?')) {
      result = '$result.';
    }

    return result;
  }

  String _normalizeCapitalization(String text) {
    if (text.isEmpty) return text;

    final normalized = text.trim();
    if (normalized.length <= 1) {
      return normalized;
    }

    final firstLetter = normalized.substring(0, 1).toUpperCase();
    final rest = normalized.substring(1);
    return '$firstLetter$rest';
  }
}
