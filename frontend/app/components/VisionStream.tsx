"use client";

import { Box } from "@mui/material";
import { SyntheticEvent, useEffect, useMemo, useRef, useState } from "react";

type StreamSize = {
  width: number;
  height: number;
};

type VisionChannelLayout = {
  channel_index: number;
  feeder_index: number;
  model_index: number;
  roi: [number, number, number, number];
};

type VisionLayoutMetadata = {
  canvas: {
    width: number;
    height: number;
  };
  channel_count: number;
  channels: VisionChannelLayout[];
};

export default function VisionStream({
  streamUrl,
  layoutMetadata,
  onChannelClick,
  alt,
}: Readonly<{
  streamUrl: string | null;
  layoutMetadata: VisionLayoutMetadata | null;
  onChannelClick?: (channelIndex: number) => void;
  alt?: string;
}>) {
  const [streamSize, setStreamSize] = useState<StreamSize | null>(null);
  const [containerSize, setContainerSize] = useState<StreamSize | null>(null);
  const [hasError, setHasError] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (containerRef.current == null) {
      return;
    }

    const updateSize = () => {
      if (containerRef.current == null) {
        return;
      }

      const rect = containerRef.current.getBoundingClientRect();
      setContainerSize({
        width: rect.width,
        height: rect.height,
      });
    };

    updateSize();

    const observer = new ResizeObserver(() => updateSize());
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const displayRect = useMemo(() => {
    const source = streamSize ?? layoutMetadata?.canvas ?? null;
    if (containerSize == null || source == null || source.width <= 0 || source.height <= 0) {
      return null;
    }

    const scale = Math.min(containerSize.width / source.width, containerSize.height / source.height);
    const width = source.width * scale;
    const height = source.height * scale;

    return {
      width,
      height,
      left: (containerSize.width - width) / 2,
      top: (containerSize.height - height) / 2,
      scale,
    };
  }, [containerSize, layoutMetadata?.canvas, streamSize]);

  if (streamUrl == null || hasError) {
    return null;
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        maxWidth: "100%",
        maxHeight: "100%",
        minWidth: 0,
        minHeight: 0,
      }}
    >
      <Box
        component="img"
        src={streamUrl}
        alt={alt ?? "Vision stream"}
        onLoad={(event: SyntheticEvent<HTMLImageElement>) => {
          const { naturalWidth, naturalHeight } = event.currentTarget;
          setHasError(false);
          if (naturalWidth > 0 && naturalHeight > 0) {
            setStreamSize({ width: naturalWidth, height: naturalHeight });
          }
        }}
        onError={() => {
          setHasError(true);
        }}
        sx={{
          display: "block",
          width: "auto",
          height: "auto",
          maxWidth: "100%",
          maxHeight: "100%",
          minWidth: 0,
          minHeight: 0,
          objectFit: "contain",
          aspectRatio: streamSize
            ? `${streamSize.width} / ${streamSize.height}`
            : layoutMetadata != null
              ? `${layoutMetadata.canvas.width} / ${layoutMetadata.canvas.height}`
              : undefined,
        }}
      />
      {displayRect != null && layoutMetadata != null &&
        <Box
          sx={{
            position: "absolute",
            left: `${displayRect.left}px`,
            top: `${displayRect.top}px`,
            width: `${displayRect.width}px`,
            height: `${displayRect.height}px`,
            pointerEvents: "none",
          }}
        >
          {layoutMetadata.channels.map((channel) => {
            const [x, y, width, height] = channel.roi;

            return (
              <Box
                key={channel.channel_index}
                component="button"
                type="button"
                aria-label={`Select video channel ${channel.channel_index + 1}`}
                onClick={() => onChannelClick?.(channel.channel_index)}
                sx={{
                  position: "absolute",
                  left: `${x * displayRect.scale}px`,
                  top: `${y * displayRect.scale}px`,
                  width: `${width * displayRect.scale}px`,
                  height: `${height * displayRect.scale}px`,
                  pointerEvents: "auto",
                  cursor: "pointer",
                  backgroundColor: "transparent",
                  border: "1px solid transparent",
                  borderRadius: "10px",
                  transition: "border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease",
                  "&:hover": {
                    borderColor: "rgba(30, 136, 255, 0.55)",
                    backgroundColor: "rgba(30, 136, 255, 0.06)",
                    boxShadow: "inset 0 0 0 1px rgba(30, 136, 255, 0.22)",
                  },
                  "&:focus-visible": {
                    outline: "none",
                    borderColor: "rgba(30, 136, 255, 0.85)",
                    backgroundColor: "rgba(30, 136, 255, 0.12)",
                    boxShadow: "0 0 0 3px rgba(30, 136, 255, 0.2)",
                  },
                }}
              />
            );
          })}
        </Box>
      }
    </Box>
  );
}
