import { Box, Chip, Grid2, Typography } from "@mui/material";
import ThermostatRoundedIcon from "@mui/icons-material/ThermostatRounded";
import BoltRoundedIcon from "@mui/icons-material/BoltRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import type { ReactNode } from "react";

type DeviceMetrics = {
  name?: string | null;
  available: boolean;
  temperature_c: number | null;
  utilization_pct: number | null;
  power_w: number | null;
  total_power_w?: number | null;
  total_tops?: number | null;
  card_count?: number | null;
  product_label?: string | null;
};

type SystemMetrics = {
  timestamp: number | null;
  cpu: DeviceMetrics | null;
  npu: DeviceMetrics | null;
};

function formatNumber(value: number | null | undefined, suffix: string): string {
  if (value == null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value.toFixed(1)}${suffix}`;
}

function MetricCell({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <Box
      sx={{
        flex: "1 1 0",
        minWidth: 0,
        padding: "clamp(10px, 1.1vh, 14px) clamp(12px, 1.1vw, 18px)",
        borderRadius: "14px",
        border: "1px solid rgba(128, 153, 184, 0.16)",
        background: "linear-gradient(180deg, rgba(22, 31, 43, 0.96) 0%, rgba(12, 18, 28, 0.98) 100%)",
        display: "flex",
        flexDirection: "column",
        gap: "clamp(4px, 0.5vh, 8px)",
      }}
    >
      <Grid2 container alignItems="center" columnSpacing="8px">
        <Grid2>{icon}</Grid2>
        <Grid2>
          <Typography sx={{ color: "#97A8BA", fontSize: "11px", fontWeight: 600 }}>
            {label}
          </Typography>
        </Grid2>
      </Grid2>
      <Typography
        sx={{
          color: "#FFFFFF",
          fontFamily: "CascadiaCode",
          fontSize: "clamp(20px, 2vh, 26px)",
          fontWeight: 600,
          lineHeight: 1.1,
        }}
      >
        {value}
      </Typography>
    </Box>
  );
}

function DeviceCard({
  title,
  metrics,
}: {
  title: string;
  metrics: DeviceMetrics | null;
}) {
  const isAvailable = metrics?.available ?? false;
  const displayName = metrics?.name ?? "Unknown device";
  const boardPower = metrics?.total_power_w ?? metrics?.power_w ?? null;

  return (
    <Box
      sx={{
        flex: "1 1 0",
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: "clamp(8px, 1vh, 14px)",
      }}
    >
      <Grid2 container justifyContent="space-between" alignItems="center" columnSpacing="10px">
        <Typography
          sx={{
            color: "#FFFFFF",
            fontSize: "clamp(18px, 1.9vh, 22px)",
            fontWeight: 600,
          }}
        >
          {title}
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: "10px", minWidth: 0 }}>
          <Typography
            title={displayName}
            sx={{
              color: "#9FB0C3",
              fontSize: "12px",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              maxWidth: "clamp(120px, 20vw, 260px)",
            }}
          >
            {displayName}
          </Typography>
          <Chip
            label={isAvailable ? "Online" : "Unavailable"}
            sx={{
              color: isAvailable ? "#D9F7E9" : "#D6DDE6",
              backgroundColor: isAvailable ? "rgba(31, 145, 94, 0.22)" : "rgba(120, 136, 153, 0.16)",
              borderRadius: "999px",
              height: "24px",
              fontSize: "11px",
              fontWeight: 600,
            }}
          />
        </Box>
      </Grid2>
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          gap: "clamp(8px, 0.8vw, 12px)",
        }}
      >
        <MetricCell
          icon={<ThermostatRoundedIcon sx={{ color: "#FFB167", fontSize: "20px" }} />}
          label="Temperature"
          value={formatNumber(metrics?.temperature_c, " C")}
        />
        <MetricCell
          icon={<MemoryRoundedIcon sx={{ color: "#77B8FF", fontSize: "20px" }} />}
          label="Utilization"
          value={formatNumber(metrics?.utilization_pct, "%")}
        />
        <MetricCell
          icon={<BoltRoundedIcon sx={{ color: "#FFD56A", fontSize: "20px" }} />}
          label="Power"
          value={formatNumber(boardPower, "W")}
        />
      </Box>
    </Box>
  );
}

export default function SystemStatusBar({
  metrics,
}: {
  metrics: SystemMetrics;
}) {
  const npuTotalTops = metrics.npu?.total_tops;
  const npuTitle = npuTotalTops != null ? `NPU (${npuTotalTops} TOPS)` : "NPU";

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "row",
        gap: "clamp(10px, 1vw, 18px)",
        width: "100%",
      }}
    >
      <DeviceCard title={npuTitle} metrics={metrics.npu} />
      <DeviceCard title="CPU" metrics={metrics.cpu} />
    </Box>
  );
}
