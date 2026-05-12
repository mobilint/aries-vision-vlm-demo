import enTexts from "./i18n/en.json";
import koTexts from "./i18n/ko.json";
import jaTexts from "./i18n/ja.json";
import zhTexts from "./i18n/zh.json";

export type LanguageText = {
  appTitle: string,
  imagePanelTitle: string,
  detectionThresholdLabel: string,
  systemMetricsButton: string,
  autoLabel: string,
  systemMetricsTitle: string,
  systemMetricsDescription: string,
  systemMetricsUpdatedAt: string,
  statusConnecting: string,
  statusPreparingPromptBundle: string,
  statusWaitingForDetection: string,
  statusWaitingForManualSelection: string,
};

export type PromptBundle = {
  system_prompt: string,
  inter_prompt: string,
};

export const DEFAULT_LANGUAGE = "en";
export const AVAILABLE_LANGUAGES = ["en", "ko", "ja", "zh"] as const;

export const language_labels: Record<string, string> = {
  en: "English",
  ko: "한국어",
  ja: "日本語",
  zh: "中文",
};

export const language_texts: Record<string, LanguageText> = {
  en: enTexts,
  ko: koTexts,
  ja: jaTexts,
  zh: zhTexts,
};

export function getLanguageTexts(language: string): LanguageText {
  return language_texts[language] ?? language_texts[DEFAULT_LANGUAGE];
}

export async function loadPromptBundle(language: string): Promise<PromptBundle> {
  const locale = AVAILABLE_LANGUAGES.includes(language as typeof AVAILABLE_LANGUAGES[number])
    ? language
    : DEFAULT_LANGUAGE;

  const [systemPrompt, interPrompt] = await Promise.all([
    fetch(`/prompt-bundles/${locale}/system.txt`).then((response) => response.text()),
    fetch(`/prompt-bundles/${locale}/inter.txt`).then((response) => response.text()),
  ]);

  return {
    system_prompt: systemPrompt.trim(),
    inter_prompt: interPrompt.trim(),
  };
}
