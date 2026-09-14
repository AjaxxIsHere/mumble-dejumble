class ProcessingResult {
  final String cleanedText;
  final bool usedLocalModel;

  const ProcessingResult({
    required this.cleanedText,
    this.usedLocalModel = false,
  });
}
