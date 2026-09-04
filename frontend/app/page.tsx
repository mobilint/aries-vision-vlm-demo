"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";
import {
  Box,
  Typography,
  Grid2,
  Button,
  createTheme,
  ThemeProvider,
} from "@mui/material";
import Chat from "./components/Chat";
import LanguageSwitcher from "./components/LanguageSwitcher";
import SystemStatusBar from "./components/SystemStatusBar";
import VisionStream from "./components/VisionStream";
import { defaultLLMClient, LLMClient, LLMState } from "./components/type";
import {
  AVAILABLE_LANGUAGES,
  DEFAULT_LANGUAGE,
  DetectionCategory,
  DETECTION_CATEGORIES,
  VISION_AUTO_TRIGGER_CONFIG_BY_CATEGORY,
  VisionAutoTriggerConfig,
  getLanguageTexts,
  loadPromptBundle,
} from "./settings";

const theme = createTheme({
  typography: {
    fontFamily: "Pretendard",
  },
});

const VISION_DETECTIONS_POLL_MS = 500;
const VISION_LAYOUT_POLL_MS = 3000;
const DEFAULT_VISION_DETECTION_TRIGGER_THRESHOLD = 0.6;
const RECENT_VLM_CHANNEL_HISTORY_LIMIT = 8;
// Keep the finished VLM answer on screen at least this long before the
// detection poller is allowed to trigger the next ask.
const MIN_ANSWER_DISPLAY_MS = 1000;

const VLM_SOCKET_PORT_BY_CATEGORY: Record<DetectionCategory, number> = {
  weapon: 5000,
  fall: 5001,
};

function makeCategoryRecord<T>(factory: (category: DetectionCategory) => T): Record<DetectionCategory, T> {
  return DETECTION_CATEGORIES.reduce((acc, category) => {
    acc[category] = factory(category);
    return acc;
  }, {} as Record<DetectionCategory, T>);
}

type VisionDetection = {
  confidence: number;
  label: number;
  label_name: string;
  roi: [number, number, number, number];
};

type VisionDetectionChannel = {
  channel_index: number;
  feeder_index: number;
  model_index: number;
  category: string;
  has_detection: boolean;
  image_width: number;
  image_height: number;
  image_base64: string;
  detections: VisionDetection[];
};

type VisionDetectionResponse = {
  channels: VisionDetectionChannel[];
};

type VisionDetectionCandidate = {
  channel: VisionDetectionChannel;
  detection: VisionDetection;
  channelHistoryRank: number;
};

type BestVisionDetection = {
  channel: VisionDetectionChannel;
  detection: VisionDetection;
};

type DetectionLocationHint = {
  horizontal: string;
  vertical: string;
  combined: string;
};

type VisionLayoutChannel = {
  channel_index: number;
  feeder_index: number;
  model_index: number;
  category: string;
  roi: [number, number, number, number];
};

type VisionLayoutResponse = {
  canvas: {
    width: number;
    height: number;
  };
  channel_count: number;
  channels: VisionLayoutChannel[];
  image_layout: {
    roi: [number, number, number, number];
  }[];
};

type VlmModelState = {
  model_id: string;
  runtime_model_id?: string;
  available_models: string[];
  is_npu?: boolean;
  is_switching?: boolean;
  message?: string | null;
};

type DeviceMetrics = {
  name?: string | null;
  available: boolean;
  temperature_c: number | null;
  utilization_pct: number | null;
  power_w: number | null;
  dram_power_w?: number | null;
  power_status?: string | null;
  power_error?: string | null;
  p99_power_w?: number | null;
  max_power_w?: number | null;
  total_power_w?: number | null;
  power_samples?: number | null;
  used_mb?: number | null;
  total_mb?: number | null;
  available_mb?: number | null;
  source: string;
};

type SystemMetrics = {
  timestamp: number | null;
  cpu: DeviceMetrics | null;
  npu: DeviceMetrics | null;
  ram?: DeviceMetrics | null;
};

type SystemMetricsPayload = SystemMetrics | {
  current: SystemMetrics;
  history?: SystemMetrics[];
  sample_interval_seconds?: number;
  max_samples?: number;
};

const DEFAULT_SYSTEM_METRICS: SystemMetrics = {
  timestamp: null,
  cpu: null,
  npu: null,
  ram: null,
};

const SYSTEM_METRICS_POLL_MS = 1000;

function getDetectionLocationHint(
  roi: [number, number, number, number],
  imageWidth: number,
  imageHeight: number,
): DetectionLocationHint {
  const [x, y, width, height] = roi;
  const centerX = imageWidth > 0 ? (x + width / 2) / imageWidth : 0.5;
  const centerY = imageHeight > 0 ? (y + height / 2) / imageHeight : 0.5;

  const horizontal = centerX < 1 / 3 ? "left" : centerX > 2 / 3 ? "right" : "center";
  const vertical = centerY < 1 / 3 ? "upper" : centerY > 2 / 3 ? "lower" : "middle";
  const combined = vertical == "middle" ? `${horizontal} area` : `${vertical}-${horizontal} area`;

  return { horizontal, vertical, combined };
}

function getCertaintyWord(confidence: number): string {
  return confidence >= 0.85 ? "high" : confidence >= 0.7 ? "medium" : "low";
}

// The VLM copies any number it sees into its answer, so the user message
// carries no roi arrays, confidence floats, or channel metadata — only
// words. Spatial information reaches the model via the red boxes drawn
// on the image plus the location hints.
function formatDetectionListForVlm(
  detections: VisionDetection[],
  imageWidth: number,
  imageHeight: number,
): string {
  if (detections.length == 0) {
    return "  - none";
  }

  return detections.map((item) => {
    const locationHint = getDetectionLocationHint(item.roi, imageWidth, imageHeight);

    return [
      "  - object: " + item.label_name,
      "    certainty: " + getCertaintyWord(item.confidence),
      "    location: " + locationHint.combined,
    ].join("\n");
  }).join("\n");
}

const VLM_ANSWER_INSTRUCTION =
  "Inspect every red visual marker on the image as primary alert evidence and answer following your instructions.";

function getVlmEligibleDetections(
  detections: VisionDetection[],
  category: DetectionCategory,
  detectionThreshold: number,
): VisionDetection[] {
  return detections.filter((detection) => {
    return isDetectionEligibleForAutoTrigger(detection, category, detectionThreshold);
  });
}

function buildDetectionPrompt(
  bestDetection: BestVisionDetection,
  category: DetectionCategory,
  thresholdPassedDetections: VisionDetection[],
): string {
  const { channel, detection } = bestDetection;
  const locationHint = getDetectionLocationHint(
    detection.roi,
    channel.image_width,
    channel.image_height,
  );

  const otherDetections = thresholdPassedDetections.filter((item) => item != detection);

  const lines = [
    "detection_event:",
    `category: ${category}`,
    "trigger_detection:",
    `  object: ${detection.label_name}`,
    `  certainty: ${getCertaintyWord(detection.confidence)}`,
    `  location: ${locationHint.combined}`,
  ];

  if (otherDetections.length > 0) {
    lines.push(
      "other_red_marked_detections:",
      formatDetectionListForVlm(otherDetections, channel.image_width, channel.image_height),
    );
  }

  lines.push(VLM_ANSWER_INSTRUCTION);
  return lines.join("\n");
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load detection image for annotation."));
    image.src = src;
  });
}

async function createAnnotatedDetectionImageDataUrl(
  imageSrc: string,
  detections: VisionDetection[],
  imageWidth: number,
  imageHeight: number,
): Promise<string> {
  if (detections.length == 0 || imageWidth <= 0 || imageHeight <= 0) {
    return imageSrc;
  }

  try {
    const image = await loadImageElement(imageSrc);
    const canvas = document.createElement("canvas");
    canvas.width = imageWidth;
    canvas.height = imageHeight;

    const context = canvas.getContext("2d");
    if (context == null) {
      return imageSrc;
    }

    context.drawImage(image, 0, 0, imageWidth, imageHeight);

    const strokeWidth = Math.max(5, Math.round(Math.min(imageWidth, imageHeight) * 0.01));

    const drawDetectionBox = (detection: VisionDetection) => {
      const [x, y, width, height] = detection.roi;
      const clampedX = Math.max(0, x);
      const clampedY = Math.max(0, y);
      const clampedWidth = Math.max(0, Math.min(width, imageWidth - clampedX));
      const clampedHeight = Math.max(0, Math.min(height, imageHeight - clampedY));

      if (clampedWidth <= 0 || clampedHeight <= 0) {
        return;
      }

      context.strokeStyle = "rgba(255, 59, 48, 0.98)";
      context.lineWidth = strokeWidth;
      context.strokeRect(clampedX, clampedY, clampedWidth, clampedHeight);
    };

    context.save();
    detections.forEach(drawDetectionBox);
    context.restore();

    return canvas.toDataURL("image/jpeg", 0.92);
  } catch (error) {
    console.warn("[vision detections] failed to annotate VLM image", error);
    return imageSrc;
  }
}

function buildManualChannelPrompt(
  channel: VisionDetectionChannel,
  category: DetectionCategory,
  thresholdPassedDetections: VisionDetection[],
): string {
  return [
    "manual_channel_selection:",
    `category: ${category}`,
    "red_marked_detections:",
    formatDetectionListForVlm(
      thresholdPassedDetections,
      channel.image_width,
      channel.image_height,
    ),
    VLM_ANSWER_INSTRUCTION,
  ].join("\n");
}

function getVisionAutoTriggerConfig(category: DetectionCategory): VisionAutoTriggerConfig {
  return VISION_AUTO_TRIGGER_CONFIG_BY_CATEGORY[category] ?? {
    allowedLabelNames: null,
    useDetectionThreshold: true,
  };
}

function isDetectionEligibleForAutoTrigger(
  detection: VisionDetection,
  category: DetectionCategory,
  detectionThreshold: number,
): boolean {
  const config = getVisionAutoTriggerConfig(category);

  if (config.allowedLabelNames != null && !config.allowedLabelNames.includes(detection.label_name)) {
    return false;
  }

  if (config.minConfidence != null && detection.confidence <= config.minConfidence) {
    return false;
  }

  if (config.useDetectionThreshold && detection.confidence <= detectionThreshold) {
    return false;
  }

  return true;
}

function isDetectionCategory(value: string): value is DetectionCategory {
  return (DETECTION_CATEGORIES as readonly string[]).includes(value);
}

export default function Page() {
  const socketsRef = useRef<Record<DetectionCategory, Socket | null>>({ weapon: null, fall: null });
  const languageRef = useRef(DEFAULT_LANGUAGE);
  const promptSyncRequestIdRef = useRef<Record<DetectionCategory, number>>({ weapon: 0, fall: 0 });
  const latestVisionDetectionsRef = useRef<VisionDetectionResponse | null>(null);
  const visionLayoutRef = useRef<VisionLayoutResponse | null>(null);
  const recentVlmChannelIndicesRef = useRef<Record<DetectionCategory, number[]>>({ weapon: [], fall: [] });
  const pendingAskRef = useRef<Record<DetectionCategory, { question: string; imageSrc: string | null } | null>>({
    weapon: null,
    fall: null,
  });
  const interactionBlockedRef = useRef(false);
  const lastAnswerEndAtRef = useRef<Record<DetectionCategory, number>>({ weapon: 0, fall: 0 });

  const [isConnected, setIsConnected] = useState<Record<DetectionCategory, boolean>>({ weapon: false, fall: false });
  const [clients, setClients] = useState<Record<DetectionCategory, LLMClient>>({
    weapon: defaultLLMClient,
    fall: defaultLLMClient,
  });
  const [language, setLanguage] = useState(DEFAULT_LANGUAGE);
  const [availableVlmModels, setAvailableVlmModels] = useState<Record<DetectionCategory, string[]>>({
    weapon: [],
    fall: [],
  });
  const [currentVlmModel, setCurrentVlmModel] = useState<Record<DetectionCategory, string>>({
    weapon: "",
    fall: "",
  });
  const [isVlmModelSwitching, setIsVlmModelSwitching] = useState<Record<DetectionCategory, boolean>>({
    weapon: false,
    fall: false,
  });
  const [vlmModelStatus, setVlmModelStatus] = useState<Record<DetectionCategory, string | null>>({
    weapon: null,
    fall: null,
  });
  const [isAutoMode, setIsAutoMode] = useState(true);
  const [isPromptConfigReady, setIsPromptConfigReady] = useState<Record<DetectionCategory, boolean>>({
    weapon: false,
    fall: false,
  });
  const [isPromptConfigSyncing, setIsPromptConfigSyncing] = useState<Record<DetectionCategory, boolean>>({
    weapon: false,
    fall: false,
  });
  const [visionStreamUrl, setVisionStreamUrl] = useState<string | null>(null);
  const [visionDetectionsUrl, setVisionDetectionsUrl] = useState<string | null>(null);
  const [visionLayoutUrl, setVisionLayoutUrl] = useState<string | null>(null);
  const [visionLayout, setVisionLayout] = useState<VisionLayoutResponse | null>(null);
  const [highlightedVisionChannelIndices, setHighlightedVisionChannelIndices] = useState<number[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>(DEFAULT_SYSTEM_METRICS);
  const [weaponThreshold, setWeaponThreshold] = useState(DEFAULT_VISION_DETECTION_TRIGGER_THRESHOLD);
  const [fallThreshold, setFallThreshold] = useState(DEFAULT_VISION_DETECTION_TRIGGER_THRESHOLD);

  const thresholds = useMemo<Record<DetectionCategory, number>>(
    () => ({ weapon: weaponThreshold, fall: fallThreshold }),
    [weaponThreshold, fallThreshold],
  );

  const texts = getLanguageTexts(language);
  const anyPromptSyncing = isPromptConfigSyncing.weapon || isPromptConfigSyncing.fall;
  const anyVlmSwitching = isVlmModelSwitching.weapon || isVlmModelSwitching.fall;
  const isInteractionBlocked = anyPromptSyncing || anyVlmSwitching;
  const allPromptConfigReady = isPromptConfigReady.weapon && isPromptConfigReady.fall;
  const allClientsIdle = clients.weapon.state == LLMState.IDLE && clients.fall.state == LLMState.IDLE;
  const allConnected = isConnected.weapon && isConnected.fall;
  const anyConnected = isConnected.weapon || isConnected.fall;

  useEffect(() => {
    interactionBlockedRef.current = isInteractionBlocked;
  }, [isInteractionBlocked]);

  useEffect(() => {
    languageRef.current = language;
    setClients((prev) => ({
      weapon: { ...prev.weapon, language },
      fall: { ...prev.fall, language },
    }));
  }, [language]);

  useEffect(() => {
    visionLayoutRef.current = visionLayout;
  }, [visionLayout]);

  useEffect(() => {
    const baseUrl = `${window.location.protocol}//${window.location.hostname}:8081`;
    setVisionStreamUrl(`${baseUrl}/stream.mjpg`);
    setVisionDetectionsUrl(`${baseUrl}/detections`);
    setVisionLayoutUrl(`${baseUrl}/layout`);
  }, []);

  useEffect(() => {
    if (visionLayoutUrl == null) {
      return;
    }

    let isCancelled = false;

    async function pollVisionLayout() {
      try {
        const layoutUrl = visionLayoutUrl;
        if (layoutUrl == null) {
          return;
        }

        const response = await fetch(layoutUrl, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Vision layout fetch failed: ${response.status}`);
        }

        const payload: VisionLayoutResponse = await response.json();
        if (!isCancelled) {
          setVisionLayout(payload);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error("[vision layout]", error);
        }
      }
    }

    pollVisionLayout();
    const intervalId = window.setInterval(pollVisionLayout, VISION_LAYOUT_POLL_MS);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [visionLayoutUrl]);

  async function loadVisionDetectionsSnapshot(): Promise<VisionDetectionResponse | null> {
    const detectionsUrl = visionDetectionsUrl;
    if (detectionsUrl == null) {
      return null;
    }

    const response = await fetch(detectionsUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Vision detections fetch failed: ${response.status}`);
    }

    const payload: VisionDetectionResponse = await response.json();
    latestVisionDetectionsRef.current = payload;
    return payload;
  }

  function updateRecentVlmChannelHistory(category: DetectionCategory, channelIndex: number) {
    const previous = recentVlmChannelIndicesRef.current[category];
    recentVlmChannelIndicesRef.current = {
      ...recentVlmChannelIndicesRef.current,
      [category]: [
        channelIndex,
        ...previous.filter((item) => item != channelIndex),
      ].slice(0, RECENT_VLM_CHANNEL_HISTORY_LIMIT),
    };
  }

  useEffect(() => {
    if (visionDetectionsUrl == null || !allConnected || !allPromptConfigReady) {
      return;
    }

    let isCancelled = false;

    function getChannelHistoryRank(category: DetectionCategory, channelIndex: number): number {
      const recent = recentVlmChannelIndicesRef.current[category];
      const recentIndex = recent.indexOf(channelIndex);
      return recentIndex == -1 ? RECENT_VLM_CHANNEL_HISTORY_LIMIT : recentIndex;
    }

    function getBestVisionDetection(
      payload: VisionDetectionResponse,
      category: DetectionCategory,
    ): BestVisionDetection | null {
      const candidates: VisionDetectionCandidate[] = [];
      const threshold = thresholds[category];

      for (const channel of payload.channels) {
        if (channel.category != category) {
          continue;
        }

        let bestDetectionInChannel: VisionDetection | null = null;

        for (const detection of channel.detections) {
          if (!isDetectionEligibleForAutoTrigger(detection, category, threshold)) {
            continue;
          }

          if (bestDetectionInChannel == null || detection.confidence > bestDetectionInChannel.confidence) {
            bestDetectionInChannel = detection;
          }
        }

        if (bestDetectionInChannel != null) {
          candidates.push({
            channel,
            detection: bestDetectionInChannel,
            channelHistoryRank: getChannelHistoryRank(category, channel.channel_index),
          });
        }
      }

      if (candidates.length == 0) {
        return null;
      }

      candidates.sort((left, right) => {
        if (left.channelHistoryRank != right.channelHistoryRank) {
          return right.channelHistoryRank - left.channelHistoryRank;
        }

        return right.detection.confidence - left.detection.confidence;
      });

      const best = candidates[0];
      return { channel: best.channel, detection: best.detection };
    }

    function getEligibleChannelIndices(payload: VisionDetectionResponse): number[] {
      const eligible = new Set<number>();
      for (const channel of payload.channels) {
        if (!isDetectionCategory(channel.category)) {
          continue;
        }
        const threshold = thresholds[channel.category];
        const hit = channel.detections.some((detection) => {
          return isDetectionEligibleForAutoTrigger(detection, channel.category as DetectionCategory, threshold);
        });
        if (hit) {
          eligible.add(channel.channel_index);
        }
      }
      return [...eligible];
    }

    async function fireAskForCategory(
      category: DetectionCategory,
      bestDetection: BestVisionDetection,
    ) {
      if (!isAutoMode || interactionBlockedRef.current) {
        return;
      }
      if (clients[category].state != LLMState.IDLE) {
        return;
      }
      if (Date.now() - lastAnswerEndAtRef.current[category] < MIN_ANSWER_DISPLAY_MS) {
        return;
      }

      updateRecentVlmChannelHistory(category, bestDetection.channel.channel_index);

      const threshold = thresholds[category];
      const selectedImage = `data:image/jpeg;base64,${bestDetection.channel.image_base64}`;
      const annotatedDetections = getVlmEligibleDetections(
        bestDetection.channel.detections,
        category,
        threshold,
      );
      const annotatedImage = await createAnnotatedDetectionImageDataUrl(
        selectedImage,
        annotatedDetections,
        bestDetection.channel.image_width,
        bestDetection.channel.image_height,
      );
      console.log("[vision detections] selected for VLM", {
        category,
        channel_index: bestDetection.channel.channel_index,
        feeder_index: bestDetection.channel.feeder_index,
        model_index: bestDetection.channel.model_index,
        confidence: bestDetection.detection.confidence,
        label: bestDetection.detection.label,
        label_name: bestDetection.detection.label_name,
        roi: bestDetection.detection.roi,
        recent_channel_history: recentVlmChannelIndicesRef.current[category],
      });
      ask(
        category,
        buildDetectionPrompt(bestDetection, category, annotatedDetections),
        annotatedImage,
      );
    }

    async function pollVisionDetections() {
      try {
        const payload = await loadVisionDetectionsSnapshot();
        if (isCancelled || payload == null) {
          return;
        }

        setHighlightedVisionChannelIndices(getEligibleChannelIndices(payload));

        for (const category of DETECTION_CATEGORIES) {
          const bestDetection = getBestVisionDetection(payload, category);
          if (bestDetection == null) {
            continue;
          }
          await fireAskForCategory(category, bestDetection);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error("[vision detections]", error);
        }
      }
    }

    pollVisionDetections();
    const intervalId = window.setInterval(pollVisionDetections, VISION_DETECTIONS_POLL_MS);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [
    visionDetectionsUrl,
    allConnected,
    allPromptConfigReady,
    clients.weapon.state,
    clients.fall.state,
    thresholds,
    isAutoMode,
    isInteractionBlocked,
  ]);

  function makeCategoryHandlers(category: DetectionCategory) {
    function onConnect() {
      setIsConnected((prev) => ({ ...prev, [category]: true }));
      setIsPromptConfigReady((prev) => ({ ...prev, [category]: false }));
      setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: true }));
    }

    function onDisconnect() {
      setIsConnected((prev) => ({ ...prev, [category]: false }));
      setIsPromptConfigReady((prev) => ({ ...prev, [category]: false }));
      setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: false }));
    }

    function onModel(model: string) {
      setClients((prev) => ({
        ...prev,
        [category]: { ...prev[category], model_id: model },
      }));
      setCurrentVlmModel((prev) => ({ ...prev, [category]: model }));
    }

    function onTasks(tasks: number) {
      setClients((prev) => ({
        ...prev,
        [category]: { ...prev[category], tasksNum: tasks },
      }));
    }

    function onStart() {
      setClients((prev) => {
        const current = prev[category];
        if (current.state == LLMState.ABORTING) {
          return prev;
        }
        return { ...prev, [category]: { ...current, state: LLMState.ANSWERING } };
      });
    }

    function onToken(token: string) {
      setClients((prev) => {
        const current = prev[category];
        if (current.state != LLMState.ANSWERING || current.dialog.length == 0) {
          return prev;
        }
        return {
          ...prev,
          [category]: {
            ...current,
            recentAnswer: current.recentAnswer == null ? token : current.recentAnswer + token,
          },
        };
      });
    }

    function onEnd(isAborted: boolean) {
      lastAnswerEndAtRef.current = {
        ...lastAnswerEndAtRef.current,
        [category]: Date.now(),
      };
      setClients((prev) => {
        const current = prev[category];
        if (current.state == LLMState.IDLE) {
          return prev;
        }

        if (isAborted && pendingAskRef.current[category] != null && current.state == LLMState.ASKING) {
          return prev;
        }

        if (current.dialog.length == 0) {
          return {
            ...prev,
            [category]: {
              ...current,
              state: LLMState.IDLE,
              recentAnswer: null,
              tasksNum: 0,
            },
          };
        }

        const newDialog = [...current.dialog];
        const recentAnswer = current.recentAnswer ?? "";
        const lastIndex = newDialog.length - 1;
        newDialog[lastIndex] = {
          ...newDialog[lastIndex],
          answer: recentAnswer + (isAborted ? " [ABORTED]" : ""),
        };

        return {
          ...prev,
          [category]: {
            ...current,
            dialog: newDialog,
            state: LLMState.IDLE,
            recentAnswer: null,
          },
        };
      });
    }

    function onResetDone() {
      const socket = socketsRef.current[category];
      const pendingAsk = pendingAskRef.current[category];
      if (socket == null || pendingAsk == null || interactionBlockedRef.current) {
        pendingAskRef.current = { ...pendingAskRef.current, [category]: null };
        if (pendingAsk != null) {
          setClients((prev) => ({
            ...prev,
            [category]: {
              ...prev[category],
              dialog: [],
              state: LLMState.IDLE,
              tasksNum: 0,
              recentAnswer: null,
            },
          }));
        }
        return;
      }

      pendingAskRef.current = { ...pendingAskRef.current, [category]: null };
      socket.emit("ask", pendingAsk.question, pendingAsk.imageSrc);
    }

    function onPromptConfigState(payload: { is_ready: boolean }) {
      setIsPromptConfigReady((prev) => ({ ...prev, [category]: payload.is_ready }));
      if (payload.is_ready) {
        setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: false }));
      }
    }

    function onPromptConfigSaved() {
      setIsPromptConfigReady((prev) => ({ ...prev, [category]: true }));
      setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: false }));
    }

    function onVlmModelState(payload: VlmModelState) {
      const currentTexts = getLanguageTexts(languageRef.current);

      if (Array.isArray(payload.available_models)) {
        setAvailableVlmModels((prev) => {
          const next = { ...prev, [category]: payload.available_models };
          const other: DetectionCategory = category == "weapon" ? "fall" : "weapon";
          const otherModels = next[other];
          if (
            otherModels.length > 0 &&
            (otherModels.length != payload.available_models.length ||
              otherModels.some((model, index) => model != payload.available_models[index]))
          ) {
            console.warn(
              "[vlm_model_state] available_models diverge between sockets",
              { weapon: next.weapon, fall: next.fall },
            );
          }
          return next;
        });
      }

      if (payload.model_id) {
        setCurrentVlmModel((prev) => ({ ...prev, [category]: payload.model_id }));
        setClients((prev) => ({
          ...prev,
          [category]: { ...prev[category], model_id: payload.model_id },
        }));
      }

      setIsVlmModelSwitching((prev) => ({ ...prev, [category]: Boolean(payload.is_switching) }));
      if (payload.is_switching) {
        setVlmModelStatus((prev) => ({ ...prev, [category]: currentTexts.vlmModelSwitchingStatus }));
      } else if (payload.message) {
        setVlmModelStatus((prev) => ({ ...prev, [category]: payload.message ?? null }));
      } else {
        setVlmModelStatus((prev) => ({ ...prev, [category]: null }));
      }
    }

    return {
      onConnect,
      onDisconnect,
      onModel,
      onTasks,
      onStart,
      onToken,
      onEnd,
      onResetDone,
      onPromptConfigState,
      onPromptConfigSaved,
      onVlmModelState,
    };
  }

  function isSystemMetricsEnvelope(payload: SystemMetricsPayload): payload is Exclude<SystemMetricsPayload, SystemMetrics> {
    return "current" in payload;
  }

  function onSystemMetrics(payload: SystemMetricsPayload) {
    const currentMetrics = isSystemMetricsEnvelope(payload) ? payload.current : payload;
    setSystemMetrics(currentMetrics);
  }

  useEffect(() => {
    const wsScheme = window.location.protocol == "https:" ? "wss" : "ws";
    const hostname = window.location.hostname;

    const cleanups: Array<() => void> = [];

    for (const category of DETECTION_CATEGORIES) {
      const port = VLM_SOCKET_PORT_BY_CATEGORY[category];
      const socket = io(`${wsScheme}://${hostname}:${port}`);
      socketsRef.current[category] = socket;

      const handlers = makeCategoryHandlers(category);
      socket.on("connect", handlers.onConnect);
      socket.on("disconnect", handlers.onDisconnect);
      socket.on("model", handlers.onModel);
      socket.on("tasks", handlers.onTasks);
      socket.on("start", handlers.onStart);
      socket.on("token", handlers.onToken);
      socket.on("end", handlers.onEnd);
      socket.on("reset_done", handlers.onResetDone);
      socket.on("prompt_config_state", handlers.onPromptConfigState);
      socket.on("prompt_config_saved", handlers.onPromptConfigSaved);
      socket.on("vlm_model_state", handlers.onVlmModelState);

      if (category == "weapon") {
        socket.on("system_metrics", onSystemMetrics);
      }

      cleanups.push(() => {
        socket.off("connect", handlers.onConnect);
        socket.off("disconnect", handlers.onDisconnect);
        socket.off("model", handlers.onModel);
        socket.off("tasks", handlers.onTasks);
        socket.off("start", handlers.onStart);
        socket.off("token", handlers.onToken);
        socket.off("end", handlers.onEnd);
        socket.off("reset_done", handlers.onResetDone);
        socket.off("prompt_config_state", handlers.onPromptConfigState);
        socket.off("prompt_config_saved", handlers.onPromptConfigSaved);
        socket.off("vlm_model_state", handlers.onVlmModelState);
        if (category == "weapon") {
          socket.off("system_metrics", onSystemMetrics);
        }
        socket.disconnect();
        socketsRef.current[category] = null;
      });
    }

    return () => {
      for (const cleanup of cleanups) {
        cleanup();
      }
    };
  }, []);

  useEffect(() => {
    for (const category of DETECTION_CATEGORIES) {
      const socket = socketsRef.current[category];
      if (isConnected[category] && socket != null) {
        socket.emit("vlm_models:get");
      }
    }
  }, [isConnected.weapon, isConnected.fall]);

  useEffect(() => {
    let isCancelled = false;

    async function syncCategoryPromptBundle(category: DetectionCategory) {
      const socket = socketsRef.current[category];
      if (socket == null || !isConnected[category]) {
        return;
      }

      promptSyncRequestIdRef.current = {
        ...promptSyncRequestIdRef.current,
        [category]: promptSyncRequestIdRef.current[category] + 1,
      };
      const requestId = promptSyncRequestIdRef.current[category];
      setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: true }));
      setIsPromptConfigReady((prev) => ({ ...prev, [category]: false }));
      try {
        const promptBundle = await loadPromptBundle(category, language);
        if (isCancelled || promptSyncRequestIdRef.current[category] != requestId) {
          return;
        }
        socketsRef.current[category]?.emit("prompt_config", promptBundle);
      } catch (error) {
        if (!isCancelled && promptSyncRequestIdRef.current[category] == requestId) {
          console.error(`[prompt bundle:${category}]`, error);
          setIsPromptConfigSyncing((prev) => ({ ...prev, [category]: false }));
        }
      }
    }

    (async () => {
      await Promise.all(DETECTION_CATEGORIES.map(syncCategoryPromptBundle));
    })();

    return () => {
      isCancelled = true;
    };
  }, [isConnected.weapon, isConnected.fall, language]);

  useEffect(() => {
    const socket = socketsRef.current.weapon;
    if (!isConnected.weapon || socket == null) {
      return;
    }

    socket.emit("system_metrics:get");
    const intervalId = window.setInterval(() => {
      socketsRef.current.weapon?.emit("system_metrics:get");
    }, SYSTEM_METRICS_POLL_MS);

    return () => window.clearInterval(intervalId);
  }, [isConnected.weapon]);

  function ask(
    category: DetectionCategory,
    newQuestion: string,
    imageSrc?: string | null,
  ) {
    const socket = socketsRef.current[category];
    if (
      socket == null ||
      newQuestion == "" ||
      !isPromptConfigReady[category] ||
      interactionBlockedRef.current
    ) {
      return;
    }

    pendingAskRef.current = {
      ...pendingAskRef.current,
      [category]: { question: newQuestion, imageSrc: imageSrc ?? null },
    };

    socket.emit("reset");

    setClients((prev) => ({
      ...prev,
      [category]: {
        ...prev[category],
        dialog: [{ question: newQuestion, answer: null }],
        image: imageSrc ?? prev[category].image,
        state: LLMState.ASKING,
        tasksNum: 0,
        recentAnswer: null,
      },
    }));
  }

  async function handleManualChannelSelect(channelIndex: number) {
    if (interactionBlockedRef.current) {
      return;
    }

    const layout = visionLayoutRef.current;
    const layoutChannel = layout?.channels.find((item) => item.channel_index == channelIndex);
    if (layoutChannel == null || !isDetectionCategory(layoutChannel.category)) {
      console.warn("[vision manual] unknown category for channel", channelIndex, layoutChannel?.category);
      return;
    }

    const category = layoutChannel.category;
    if (!isConnected[category]) {
      console.warn("[vision manual] socket not connected for category", category);
      return;
    }
    const socket = socketsRef.current[category];
    if (socket == null || !isPromptConfigReady[category]) {
      return;
    }

    setIsAutoMode(false);
    setHighlightedVisionChannelIndices([channelIndex]);

    try {
      const payload = await loadVisionDetectionsSnapshot() ?? latestVisionDetectionsRef.current;
      const channel = payload?.channels.find((item) => item.channel_index == channelIndex);
      if (channel == null || channel.image_base64 == "") {
        console.warn("[vision manual] no latest frame for channel", channelIndex);
        return;
      }

      const threshold = thresholds[category];
      const thresholdPassedDetections = getVlmEligibleDetections(
        channel.detections,
        category,
        threshold,
      );

      const selectedImage = `data:image/jpeg;base64,${channel.image_base64}`;
      const annotatedImage = await createAnnotatedDetectionImageDataUrl(
        selectedImage,
        thresholdPassedDetections,
        channel.image_width,
        channel.image_height,
      );

      ask(
        category,
        buildManualChannelPrompt(channel, category, thresholdPassedDetections),
        annotatedImage,
      );
    } catch (error) {
      console.error("[vision manual]", error);
    }
  }

  function reset() {
    pendingAskRef.current = { weapon: null, fall: null };
    setHighlightedVisionChannelIndices([]);

    setClients((prev) => makeCategoryRecord((category) => ({
      ...prev[category],
      image: null,
      dialog: [],
      recentAnswer: null,
      tasksNum: 0,
      state: LLMState.IDLE,
    })));

    for (const category of DETECTION_CATEGORIES) {
      socketsRef.current[category]?.emit("reset");
    }
  }

  function changeLanguage(nextLanguage: string) {
    if (nextLanguage == language || isInteractionBlocked) {
      return;
    }

    setIsPromptConfigSyncing({ weapon: true, fall: true });
    reset();
    setLanguage(nextLanguage);
  }

  function changeVlmModel(category: DetectionCategory, nextModel: string) {
    const socket = socketsRef.current[category];
    if (
      socket == null ||
      nextModel == currentVlmModel[category] ||
      isInteractionBlocked
    ) {
      return;
    }

    setIsVlmModelSwitching((prev) => ({ ...prev, [category]: true }));
    setVlmModelStatus((prev) => ({
      ...prev,
      [category]: getLanguageTexts(languageRef.current).vlmModelSwitchingStatus,
    }));
    reset();
    socket.emit("vlm_model:set", { model_id: nextModel });
  }

  function enableAutoMode() {
    if (!allPromptConfigReady || !allClientsIdle || isInteractionBlocked) {
      return;
    }

    setIsAutoMode(true);
    setHighlightedVisionChannelIndices([]);
  }

  const categoryLabels: Record<DetectionCategory, string> = {
    weapon: texts.weaponPanelTitle,
    fall: texts.fallPanelTitle,
  };
  const combinedVlmModelStatus: string | null = (() => {
    const weaponStatus = vlmModelStatus.weapon;
    const fallStatus = vlmModelStatus.fall;
    if (weaponStatus != null && fallStatus != null) {
      return `${categoryLabels.weapon}: ${weaponStatus}\n${categoryLabels.fall}: ${fallStatus}`;
    }
    if (weaponStatus != null) {
      return weaponStatus;
    }
    if (fallStatus != null) {
      return fallStatus;
    }
    return null;
  })();

  if (!anyConnected) {
    return (
      <ThemeProvider theme={theme}>
        <Grid2
          container
          justifyContent="center"
          alignItems="center"
          sx={{ width: "100vw", height: "100vh", backgroundColor: "#111111" }}
        >
          <Typography sx={{ color: "#FFFFFF", fontSize: "24px", fontWeight: 500 }}>
            {texts.statusConnecting}
          </Typography>
        </Grid2>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          width: "100vw",
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "auto",
          backgroundColor: "#000000",
          padding: "62px 36px 68px 36px",
        }}
      >
        <Grid2
          container
          direction="column"
          wrap="nowrap"
          sx={{
            width: "100%",
            height: "100%",
            color: "#FFFFFF",
            overflow: "hidden",
          }}
        >
          <Grid2
            container
            justifyContent="space-between"
            alignItems="center"
            wrap="nowrap"
            padding="0"
          >
            <Grid2 container alignItems="center" wrap="nowrap" columnSpacing="26px" sx={{ minWidth: 0 }}>
              <Box
                component="img"
                src="/goodway_logo.png"
                alt="Goodway"
                sx={{ height: "60px", width: "auto", display: "block" }}
              />
              <Typography
                sx={{
                  color: "#FFFFFF",
                  fontSize: "48px",
                  fontWeight: 500,
                  letterSpacing: "-1%",
                  lineHeight: "120%",
                  whiteSpace: "nowrap",
                }}
              >
                {texts.appTitle}
              </Typography>
            </Grid2>
            <Grid2
              container
              alignItems="center"
              columnSpacing="16px"
              sx={{ width: "fit-content" }}
            >
              <Button
                disableElevation
                disabled={!allPromptConfigReady || !allClientsIdle || isInteractionBlocked}
                onClick={enableAutoMode}
                sx={{
                  padding: "15px 23px",
                  borderRadius: "999px",
                  fontWeight: 500,
                  fontSize: "14.71px",
                  border: "none",
                  color: "#FFFFFF",
                  backgroundColor: isAutoMode ? "#0B6BFF" : "#2A2A2A",
                  boxShadow: "none",
                  "&:hover": {
                    backgroundColor: isAutoMode ? "#0B6BFF" : "#343434",
                  },
                  "&.Mui-disabled": {
                    color: "#A0A0A0",
                    backgroundColor: isAutoMode ? "#164C9F" : "#242424",
                  },
                }}
              >
                {texts.autoLabel}
              </Button>
              <LanguageSwitcher
                languages={[...AVAILABLE_LANGUAGES]}
                currentLanguage={language}
                disabled={isInteractionBlocked}
                changeLanguage={changeLanguage}
              />
            </Grid2>
          </Grid2>
          <Box sx={{ flex: "0 0 1px", width: "100%", backgroundColor: "rgba(255,255,255,0.36)", mt: "35px", mb: "69px" }} />
          <Grid2
            container
            direction="row"
            size="grow"
            columnSpacing="24px"
            wrap="nowrap"
            sx={{ minHeight: 0 }}
          >
            <Grid2
              container
              alignSelf="flex-start"
              sx={{
                height: "100%",
                minHeight: 0,
                minWidth: 0,
                flex: "0 0 auto",
              }}
            >
              <VisionStream
                streamUrl={visionStreamUrl}
                layoutMetadata={visionLayout}
                highlightedChannelIndices={highlightedVisionChannelIndices}
                onChannelClick={handleManualChannelSelect}
                disabled={isInteractionBlocked}
                alt="32-channel vision stream"
              />
            </Grid2>
            <Grid2
              container
              direction="column"
              wrap="nowrap"
              rowSpacing="clamp(16px, 1.6vh, 26px)"
              sx={{
                minHeight: 0,
                minWidth: 0,
                flex: "1 1 auto",
              }}
            >
              <Grid2 sx={{ flex: "0 0 auto", width: "100%" }}>
                <SystemStatusBar metrics={systemMetrics} />
              </Grid2>
              <Grid2 sx={{ flex: "1 1 auto", minHeight: 0, width: "100%" }}>
                <Chat
                  clients={clients}
                  language={language}
                  isAutoMode={isAutoMode}
                  isConnected={isConnected}
                  thresholds={thresholds}
                  setThreshold={{ weapon: setWeaponThreshold, fall: setFallThreshold }}
                  vlmModels={availableVlmModels}
                  currentVlmModel={{
                    weapon: currentVlmModel.weapon || clients.weapon.model_id,
                    fall: currentVlmModel.fall || clients.fall.model_id,
                  }}
                  isVlmModelSwitching={isVlmModelSwitching}
                  isInteractionBlocked={isInteractionBlocked}
                  changeVlmModel={changeVlmModel}
                />
              </Grid2>
            </Grid2>
          </Grid2>
          {(combinedVlmModelStatus != null || !allPromptConfigReady) &&
            <Typography
              sx={{
                color: "#A8A8A8",
                fontSize: "14px",
                lineHeight: "130%",
                whiteSpace: "pre-line",
              }}
            >
              {combinedVlmModelStatus ?? texts.statusPreparingPromptBundle}
            </Typography>
          }
        </Grid2>
      </Box>
    </ThemeProvider>
  );
}
