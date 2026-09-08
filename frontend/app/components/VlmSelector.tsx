import { Button, ListItemText, Menu, MenuItem } from "@mui/material";
import ExpandMoreRoundedIcon from "@mui/icons-material/ExpandMoreRounded";
import { useState } from "react";
import { LanguageText, getVlmModelLabel } from "../settings";

export default function VlmSelector({
  models,
  currentModel,
  disabled,
  texts,
  changeModel,
}: {
  models: string[],
  currentModel: string,
  disabled: boolean,
  texts: Pick<LanguageText, "vlmModelLabel">,
  changeModel: (modelId: string) => void,
}) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const isOpen = anchorEl != null;

  return (
    <>
      <Button
        disableElevation
        disabled={disabled || models.length == 0}
        endIcon={<ExpandMoreRoundedIcon sx={{ fontSize: "14px" }} />}
        onClick={(event) => setAnchorEl(event.currentTarget)}
        sx={{
          padding: "15px 23px",
          borderRadius: "999px",
          color: "#FFFFFF",
          backgroundColor: "#2A2A2A",
          border: "none",
          fontSize: "12px",
          fontWeight: 500,
          whiteSpace: "nowrap",
          textTransform: "none",
          "&:hover": { backgroundColor: "#343434" },
          "&.Mui-disabled": {
            color: "#A0A0A0",
            backgroundColor: "#242424",
          },
        }}
      >
        {currentModel ? getVlmModelLabel(currentModel) : texts.vlmModelLabel}
      </Button>
      <Menu
        anchorEl={anchorEl}
        open={isOpen}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: {
              marginTop: "10px",
              borderRadius: "16px",
              border: "1px solid #363636",
              backgroundColor: "#1F1F1F",
              minWidth: "260px",
              overflow: "hidden",
            },
          },
        }}
      >
        {models.map((modelId) => {
          const isActive = currentModel == modelId;
          return (
            <MenuItem
              key={modelId}
              selected={isActive}
              onClick={() => {
                setAnchorEl(null);
                if (modelId != currentModel) changeModel(modelId);
              }}
              sx={{ minHeight: "44px", backgroundColor: isActive ? "#242424" : "#181818" }}
            >
              <ListItemText
                primary={getVlmModelLabel(modelId)}
                secondary={modelId}
                primaryTypographyProps={{
                  fontSize: "11px",
                  fontWeight: isActive ? 700 : 500,
                  color: "#F2F2F2",
                }}
                secondaryTypographyProps={{
                  fontSize: "9px",
                  fontWeight: 600,
                  letterSpacing: "0.02em",
                  color: "#8F8F8F",
                }}
              />
            </MenuItem>
          );
        })}
      </Menu>
    </>
  );
}