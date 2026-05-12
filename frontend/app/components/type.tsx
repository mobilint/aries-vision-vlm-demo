export type QNA = {
  question: string,
  answer: null | string,
};

export type ImageDetectionOverlay = {
  roi: [number, number, number, number],
  imageWidth: number,
  imageHeight: number,
};

export type DialogType = QNA[];

export enum LLMState {
  IDLE,
  ASKING,
  ANSWERING,
  ABORTING,
};

export type LLMClient = {
  model_id: string,
  tasksNum: number,
  state: LLMState,
  dialog: DialogType,
  recentAnswer: string | null,
  image: null | string,
  imageDetectionOverlay: ImageDetectionOverlay | null,
  language: string,
};

export const defaultLLMClient: LLMClient = {
  model_id: "",
  tasksNum: 0,
  state: LLMState.IDLE,
  dialog: [],
  recentAnswer: null,
  image: null,
  imageDetectionOverlay: null,
  language: "en",
};
