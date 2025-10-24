# FunGen Rewrite - Keyboard Shortcuts Reference

**Version:** 1.0
**Date:** 2025-10-24
**Author:** ui-enhancer agent

---

## Overview

FunGen Rewrite features comprehensive keyboard shortcuts for efficient workflow. All shortcuts are designed to be intuitive and follow common application conventions.

---

## Global Shortcuts

### File Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+O` | Open Video | Open single video file for processing |
| `Ctrl+Shift+O` | Open Folder | Open folder containing multiple videos |
| `Ctrl+S` | Save Settings | Save current configuration |
| `Ctrl+,` | Open Settings | Open settings panel |
| `Ctrl+Q` | Quit Application | Exit FunGen (prompts if processing) |

### Processing Controls
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Space` | Play/Pause | Toggle video processing |
| `Ctrl+R` | Start Processing | Begin processing selected video(s) |
| `Ctrl+.` | Stop Processing | Stop current processing |
| `Ctrl+P` | Pause Processing | Pause processing (resume with Space) |

### View Controls
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+T` | Toggle Theme | Switch between dark/light themes |
| `Ctrl+D` | Toggle Dashboard | Show/hide agent dashboard |
| `Ctrl+F` | Toggle Fullscreen | Enter/exit fullscreen mode |
| `Ctrl+0` | Reset Zoom | Reset video preview zoom to 100% |
| `Ctrl++` | Zoom In | Increase video preview zoom |
| `Ctrl+-` | Zoom Out | Decrease video preview zoom |

### Navigation
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Tab` | Next Field | Move focus to next input field |
| `Shift+Tab` | Previous Field | Move focus to previous field |
| `Ctrl+Tab` | Next Tab | Switch to next settings tab |
| `Ctrl+Shift+Tab` | Previous Tab | Switch to previous settings tab |

### Video Preview
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Left Arrow` | Previous Frame | Step backward one frame |
| `Right Arrow` | Next Frame | Step forward one frame |
| `Ctrl+Left` | Skip Backward | Jump backward 5 seconds |
| `Ctrl+Right` | Skip Forward | Jump forward 5 seconds |
| `Home` | Go to Start | Jump to video beginning |
| `End` | Go to End | Jump to video end |
| `M` | Toggle Mute | Mute/unmute preview audio |

---

## Settings Panel Shortcuts

### Tracker Configuration
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Alt+1` | ByteTrack | Select ByteTrack algorithm |
| `Alt+2` | BoT-SORT | Select BoT-SORT algorithm |
| `Alt+3` | Hybrid Tracker | Select Hybrid algorithm |
| `Ctrl+B` | Adjust Batch Size | Focus batch size input |

### Hardware Settings
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+H` | Toggle HW Accel | Enable/disable hardware acceleration |
| `Ctrl+G` | GPU Selection | Open GPU selection dialog |
| `Ctrl+M` | Memory Limit | Set VRAM/memory limits |

### Output Settings
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+S` | Output Directory | Select output directory |
| `Ctrl+E` | Export Settings | Export current settings to file |
| `Ctrl+I` | Import Settings | Import settings from file |

---

## Agent Dashboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `F5` | Refresh Agents | Manually refresh agent status |
| `Ctrl+L` | View Logs | Open agent logs viewer |
| `Ctrl+K` | Clear Logs | Clear agent log history |
| `Ctrl+A` | Select All Agents | Select all agents in dashboard |

---

## Debug Mode Shortcuts

**Note:** Debug shortcuts only work when debug mode is enabled (`Ctrl+Shift+D`)

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+D` | Toggle Debug Mode | Enable/disable debug features |
| `F12` | Developer Console | Open debug console |
| `Ctrl+Shift+I` | Inspector | Open widget inspector |
| `Ctrl+Shift+P` | Performance Monitor | Show FPS/memory graph |
| `Ctrl+Shift+R` | Reload UI | Reload UI without restart |

---

## Context Menu Shortcuts

Right-click on various UI elements for context menus:

### Video List Context Menu
- **Play Video:** Double-click or `Enter`
- **Remove from List:** `Delete`
- **Open in Explorer:** `Ctrl+Shift+E`
- **Copy Path:** `Ctrl+C`

### Progress Bar Context Menu
- **Cancel Job:** `Delete`
- **View Details:** `Enter`
- **Open Output:** `Ctrl+Enter`

---

## Accessibility Shortcuts

### High Contrast Mode
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Alt+Shift+H` | High Contrast | Toggle high contrast theme |
| `Ctrl+Shift++` | Increase Font Size | Increase UI font size |
| `Ctrl+Shift+-` | Decrease Font Size | Decrease UI font size |
| `Ctrl+Shift+0` | Reset Font Size | Reset to default font size |

### Screen Reader Support
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+A` | Announce Status | Read current status aloud |
| `Ctrl+Shift+L` | Label Mode | Show labels for all controls |

---

## Advanced Features

### Batch Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Shift+A` | Add to Queue | Add current video to batch queue |
| `Ctrl+Shift+Q` | Process Queue | Start batch processing |
| `Ctrl+Shift+C` | Clear Queue | Clear batch queue |

### Funscript Editor (Future)
| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+Z` | Undo | Undo last edit |
| `Ctrl+Y` | Redo | Redo last undone edit |
| `Ctrl+X` | Cut | Cut selected points |
| `Ctrl+C` | Copy | Copy selected points |
| `Ctrl+V` | Paste | Paste points |
| `Delete` | Delete Points | Delete selected points |
| `Ctrl+A` | Select All | Select all points |

---

## Customization

Keyboard shortcuts can be customized in:
- **Settings → Keyboard Shortcuts**
- **File → Preferences → Shortcuts**

### Custom Shortcut File

Shortcuts are stored in:
```
~/.fungen/shortcuts.json
```

Example format:
```json
{
  "shortcuts": {
    "open_video": "Ctrl+O",
    "start_processing": "Ctrl+R",
    "toggle_theme": "Ctrl+T",
    "custom_action": "Ctrl+Shift+X"
  }
}
```

---

## Tips and Tricks

### Power User Tips

1. **Quick Processing:** `Ctrl+O` → Select video → `Space` to start
2. **Theme Switching:** `Ctrl+T` cycles through themes quickly
3. **Batch Processing:** Select folder with `Ctrl+Shift+O`, then `Ctrl+R`
4. **Monitor Performance:** `Ctrl+Shift+P` shows real-time metrics
5. **Quick Settings:** `Ctrl+,` opens settings instantly

### Efficiency Shortcuts

- **One-handed operation:** Most common actions use left hand (Ctrl+key)
- **Modal dialogs:** `Esc` always cancels, `Enter` confirms
- **Tab navigation:** Use `Tab` to navigate without mouse
- **Quick help:** `F1` shows context-sensitive help

### Accessibility

- All shortcuts have tooltip hints when hovering
- Shortcuts are displayed in menu items
- Screen reader announces shortcut when focused
- High contrast mode makes shortcuts more visible

---

## Platform-Specific Notes

### Windows
- Uses `Ctrl` as primary modifier
- `Alt` for alternative actions
- `Win` key reserved for system shortcuts

### Linux
- Uses `Ctrl` as primary modifier
- `Alt` and `Super` available
- Respects system keyboard layouts

### macOS (Future Support)
- Would use `Cmd` instead of `Ctrl`
- `Option` instead of `Alt`
- Standard macOS conventions

---

## Troubleshooting

### Shortcuts Not Working

1. **Check if another application is capturing shortcuts**
   - Close conflicting applications
   - Disable system-wide hotkey tools

2. **Verify keyboard layout**
   - Some shortcuts may vary with keyboard layout
   - Check Settings → Keyboard for current bindings

3. **Debug mode conflicts**
   - Debug shortcuts may override normal shortcuts
   - Disable debug mode with `Ctrl+Shift+D`

4. **Widget focus issues**
   - Some shortcuts require specific widget focus
   - Click on main window to restore focus

### Reset to Defaults

To reset all shortcuts to defaults:
1. Close FunGen
2. Delete `~/.fungen/shortcuts.json`
3. Restart FunGen

Or use: **Settings → Keyboard Shortcuts → Reset to Defaults**

---

## Future Enhancements

Planned keyboard shortcut features:

- [ ] Macro recording (record sequence of shortcuts)
- [ ] Multi-key chord shortcuts (e.g., `Ctrl+K, Ctrl+T`)
- [ ] Shortcut hints overlay (show all available shortcuts)
- [ ] Voice command integration
- [ ] Gamepad/controller support for video preview
- [ ] Customizable shortcut sets (presets for different workflows)

---

## Quick Reference Card

**Print this section for desk reference:**

```
╔════════════════════════════════════════════════════════╗
║         FunGen Rewrite - Quick Shortcuts               ║
╠════════════════════════════════════════════════════════╣
║ Ctrl+O      Open Video                                 ║
║ Ctrl+R      Start Processing                           ║
║ Space       Play/Pause                                 ║
║ Ctrl+T      Toggle Theme                               ║
║ Ctrl+,      Settings                                   ║
║ Ctrl+Q      Quit                                       ║
║ F5          Refresh Agents                             ║
║ Ctrl+D      Toggle Dashboard                           ║
║ Ctrl+Shift+P  Performance Monitor                      ║
╚════════════════════════════════════════════════════════╝
```

---

## Contact & Support

For shortcut customization help:
- GitHub: https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator
- Documentation: /home/pi/elo_elo_320/docs/
- Settings file: ~/.fungen/shortcuts.json

---

**Last Updated:** 2025-10-24
**Version:** 1.0
**Author:** ui-enhancer agent
