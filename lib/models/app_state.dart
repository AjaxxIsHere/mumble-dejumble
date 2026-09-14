enum VoiceAppState {
  idle,
  initializingStt,
  initializingLlm,
  ready,
  recording,
  transcribing,
  processing,
  completed,
  error,
}
