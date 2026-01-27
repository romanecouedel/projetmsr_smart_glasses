# Raspberry Pi 3B+ Configuration - MSR Project

## System Information

**OS**: Raspberry Pi OS Bookworm ARM64  
**Image**: `2024-03-12-raspios-bookworm-arm64.img.xz`  
**Source**: [Official Raspberry Pi Archives](https://www.raspberrypi.com/software/operating-systems/)

---

## Operating Principle

Instead of emulating a USB device, the Raspberry Pi creates its own Wi-Fi network to which the PC connects directly. This enables data exchange via:

- SSH
- IP Network
- Network MIDI
- Keyboard commands
- File sharing

---

## Step 1: Wi-Fi Access Point Configuration

### Installing Dependencies
```bash
sudo apt install hostapd dnsmasq
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq
```

### Checking and Enabling NetworkManager
```bash
systemctl status NetworkManager
```

If NetworkManager is not active:
```bash
sudo systemctl enable --now NetworkManager
```

### Creating the Wi-Fi Hotspot
```bash
sudo nmcli device wifi hotspot \
  ifname wlan0 \
  ssid RPI_DIRECT \
  password projet123
```

### Enabling Automatic Startup
```bash
nmcli connection show
nmcli connection modify Hotspot connection.autoconnect yes
```

---

## Step 2: Camera Configuration

### Physical Installation

Connect the camera to the Raspberry Pi's CAM port (blue side facing the Ethernet port).

### Verification and Installation
```bash
libcamera-hello
sudo apt install python3-opencv python3-picamera2
```

### Running the Script
```bash
python3 com_pc.py
```

*(The `com_pc.py` file can be found in the project folder)*

---

## Connection and Usage

### From the PC

1. **Connect to the Wi-Fi network**: `RPI_HOTSPOT`

2. **SSH Connection**:
```bash
   ssh projetmsr@10.42.0.1
```
   Password: `projet123`

### On the Raspberry Pi Side

**Check the Arduino IP address**:
```bash
ip a
sudo nano com_pc.py
```
Ensure the IP in the code matches the Arduino's IP.

**Run the script**:
```bash
python3 com_pc.py
```

### On the PC Side

**Check the firewall**:
```bash
sudo ufw status
```

If active, allow UDP port 5000:
```bash
sudo ufw allow 5000/udp
```

**Run your reception code (basic code to receive data from pc can be found in the project folder pc_cam_com.py)**

---

## Credentials

- **Username**: `projetmsr`
- **Password**: `projet123`
- **Wi-Fi SSID**: `RPI_DIRECT`
- **Raspberry Pi IP**: `10.42.0.1`
- **UDP Port**: `5000`