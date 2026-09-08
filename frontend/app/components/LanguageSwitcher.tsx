import { IconButton, ListItemText, Menu, MenuItem } from "@mui/material";
import PublicIcon from "@mui/icons-material/Public";
import { useState } from "react";
import { language_labels } from "../settings";

export default function LanguageSwitcher({
  languages,
  currentLanguage,
  disabled,
  changeLanguage,
}: {
  languages: string[],
  currentLanguage: string,
  disabled: boolean,
  changeLanguage: (language: string) => void,
}) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const isOpen = anchorEl != null;

  return (
    <>
      <IconButton
        disabled={disabled}
        onClick={(event) => setAnchorEl(event.currentTarget)}
        sx={{
          width: "54px",
          height: "54px",
          backgroundColor: "#2A2A2A",
          border: "1px solid rgba(255,255,255,0.08)",
          color: "#FFFFFF",
          borderRadius: "50%",
          "&:hover": {
            backgroundColor: "#343434",
          },
          "&.Mui-disabled": {
            color: "#777777",
            backgroundColor: "#242424",
            borderColor: "rgba(255,255,255,0.04)",
          },
        }}
      >
        <PublicIcon sx={{ fontSize: "24px" }} />
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        open={isOpen}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
        slotProps={{
          paper: {
            sx: {
              marginTop: "10px",
              borderRadius: "16px",
              border: "1px solid #363636",
              backgroundColor: "#1F1F1F",
              minWidth: "180px",
              overflow: "hidden",
            },
          },
        }}
      >
        {languages.map((language) => {
          const isActive = currentLanguage == language;

          return (
            <MenuItem
              key={language}
              selected={isActive}
              onClick={() => {
                setAnchorEl(null);
                if (language != currentLanguage)
                  changeLanguage(language);
              }}
              sx={{
                minHeight: "44px",
                backgroundColor: isActive ? "#242424" : "#181818",
              }}
            >
              <ListItemText
                primary={language_labels[language] ?? language.toUpperCase()}
                secondary={language.toUpperCase()}
                primaryTypographyProps={{
                  fontSize: "14px",
                  fontWeight: isActive ? 700 : 500,
                  color: "#F2F2F2",
                }}
                secondaryTypographyProps={{
                  fontSize: "11px",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
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
