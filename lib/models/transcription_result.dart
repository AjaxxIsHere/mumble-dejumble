class TranscriptionResult {
  final String rawText;
  final String? finalText;
  final DateTime timestamp;

  const TranscriptionResult({
    required this.rawText,
    this.finalText,
    required this.timestamp,
  });
}
