# GCP VM deployment for Snap Harvester live engine

This folder contains ready-to-use configs that match the deployment blueprint.

## Files

- `snap-harvester.service` – systemd unit for the live engine.
- `snap-harvester.env.example` – template for `/etc/snap_harvester.env`.
- `check_snap_health.sh` – optional watchdog script for cron.
- `logrotate_snap-harvester.conf` – logrotate config for app logs.

## Zero-to-first-trade checklist

### 1. Create VM + user (once)

On GCP:
- Create an Ubuntu 22.04 VM (e2-standard-2, 50 GB disk).
- Add your SSH key.

On the VM (as the default user):
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip tmux htop

sudo adduser --disabled-password --gecos "" snap
sudo usermod -aG sudo snap
sudo mkdir -p /home/snap/.ssh
sudo cp ~/.ssh/authorized_keys /home/snap/.ssh/
sudo chown -R snap:snap /home/snap/.ssh
sudo chmod 700 /home/snap/.ssh
sudo chmod 600 /home/snap/.ssh/authorized_keys
```

Reconnect as `snap`:
```bash
ssh snap@VM_IP
```

### 2. Clone repo and create venv

```bash
sudo mkdir -p /opt/snap_harvester_live_engine
sudo chown -R snap:snap /opt/snap_harvester_live_engine
cd /opt/snap_harvester_live_engine

git clone https://github.com/MbuguaOwen/shockflip_snap_harvester_v2.git .

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure live YAML + model

- Ensure `configs/snap_harvester_live_btc.yaml` matches what you tested locally.
- Ensure the `live.model_path` points to a real model file on the VM
  (e.g. `results/models/hgb_snap_2024_BTC_agg.joblib`).

### 4. Secrets env file

As root:
```bash
cp /opt/snap_harvester_live_engine/deploy/gcp_vm/snap-harvester.env.example /etc/snap_harvester.env
chmod 600 /etc/snap_harvester.env
chown root:root /etc/snap_harvester.env
```

Edit `/etc/snap_harvester.env` and fill:
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

### 5. Systemd service

As root:
```bash
cp /opt/snap_harvester_live_engine/deploy/gcp_vm/snap-harvester.service /etc/systemd/system/snap-harvester.service
systemctl daemon-reload
systemctl enable snap-harvester.service
systemctl start snap-harvester.service
```

### 6. Optional: health watchdog + logrotate

Health watchdog:
```bash
cp /opt/snap_harvester_live_engine/deploy/gcp_vm/check_snap_health.sh /usr/local/bin/check_snap_health.sh
chmod 755 /usr/local/bin/check_snap_health.sh
crontab -e
# add:
# */5 * * * * /usr/local/bin/check_snap_health.sh
```

Logrotate:
```bash
cp /opt/snap_harvester_live_engine/deploy/gcp_vm/logrotate_snap-harvester.conf /etc/logrotate.d/snap-harvester
```

### 7. Verify first heartbeat and trade

- Watch logs:
  ```bash
  journalctl -u snap-harvester.service -f
  ```
- Confirm CSVs are updating:
  ```bash
  ls -l /opt/snap_harvester_live_engine/results/live
  tail -f /opt/snap_harvester_live_engine/results/live/events.csv
  tail -f /opt/snap_harvester_live_engine/results/live/trades.csv
  ```

## Quick commands (day-to-day)

As `snap` on the VM:

- Restart after code/config change:
  ```bash
  cd /opt/snap_harvester_live_engine
  git pull
  source venv/bin/activate
  pip install -r requirements.txt  # if deps changed
  sudo systemctl restart snap-harvester.service
  ```

- Check status:
  ```bash
   systemctl status snap-harvester.service
   ```
- Live logs:
  ```bash
  journalctl -u snap-harvester.service -f
  ```

