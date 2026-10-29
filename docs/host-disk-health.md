# Host disk health

When ast-context-cache runs in an LXC container or a VM, the dashboard's disk card can't show SMART health: the disks belong to the hypervisor host. A container shares the host's kernel but has no access to the raw block devices, and a VM only sees virtual disks that have no real SMART data. The fix is to run an exporter on the host that reads SMART for the guest, and point the dashboard at it.

ast-context-cache uses [smartctl_exporter](https://github.com/prometheus-community/smartctl_exporter) for this. It is an existing prometheus-community project, not part of ast-context-cache: it runs `smartctl` (from smartmontools) on the host and serves the results as Prometheus metrics over HTTP. Any hypervisor works (Proxmox VE, plain KVM/libvirt, Unraid, TrueNAS, a Docker host) as long as the guest can reach the exporter over the network.

The setup is three steps: install the exporter on the host, make it reachable from the guest only, then paste its URL into **Settings → Host disks**.

## 1. Install smartctl_exporter on the host

Run these on the hypervisor host, not in the guest. The exporter needs **smartmontools 7.0 or newer** on the host, because it reads `smartctl`'s JSON output, which arrived in 7.0. On Debian-based hosts (including Proxmox VE): `apt install smartmontools`.

Pick one of the three options below. Each one sets the listen address to the host's LAN, bridge or tailnet IP; `192.168.1.10` stands for that address throughout this page.

### Option A: release binary and a systemd unit

Download the tarball for the host's architecture from the [releases page](https://github.com/prometheus-community/smartctl_exporter/releases). At the time of writing the latest release is 0.15.0, published as `smartctl_exporter-0.15.0.linux-amd64.tar.gz` (also `linux-arm64` and others), with a `sha256sums.txt` next to it.

```bash
VER=0.15.0 ARCH=linux-amd64
cd /tmp
curl -LO https://github.com/prometheus-community/smartctl_exporter/releases/download/v$VER/smartctl_exporter-$VER.$ARCH.tar.gz
curl -LO https://github.com/prometheus-community/smartctl_exporter/releases/download/v$VER/sha256sums.txt
sha256sum --check --ignore-missing sha256sums.txt
tar xzf smartctl_exporter-$VER.$ARCH.tar.gz
install -m 0755 smartctl_exporter-$VER.$ARCH/smartctl_exporter /usr/local/bin/smartctl_exporter
```

Then add `/etc/systemd/system/smartctl_exporter.service`:

```ini
[Unit]
Description=smartctl_exporter (SMART metrics for Prometheus)
Wants=network-online.target
After=network-online.target

[Service]
User=root
ExecStart=/usr/local/bin/smartctl_exporter --web.listen-address=192.168.1.10:9633
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now smartctl_exporter
```

### Option B: Docker or Compose

The project publishes an image that includes smartmontools. Its README runs it privileged and as root so it can reach the disks:

```yaml
services:
  smartctl-exporter:
    image: prometheuscommunity/smartctl-exporter
    privileged: true
    user: root
    ports:
      - "192.168.1.10:9633:9633"
    restart: unless-stopped
```

The README's example publishes `"9633:9633"` on every interface; the `192.168.1.10:` prefix above narrows it to one address (see [step 2](#2-expose-it-to-the-guest-only)). The image is also on Quay as `quay.io/prometheuscommunity/smartctl-exporter`.

### Option C: Debian package

Debian packages it as `prometheus-smartctl-exporter` (version 0.14.0 when this was written). It is in **trixie-backports**, **forky** (testing) and **sid**, not in trixie's main archive and not in bookworm. On a Debian 13 (trixie) based host, such as Proxmox VE 9, with backports enabled:

```bash
apt install -t trixie-backports prometheus-smartctl-exporter
```

The package installs `/usr/bin/smartctl_exporter` and a `smartctl_exporter.service` unit. Check how the unit starts the exporter with `systemctl cat smartctl_exporter`, and if it does not already bind to the address you want, override it with `systemctl edit smartctl_exporter`:

```ini
[Service]
ExecStart=
ExecStart=/usr/bin/smartctl_exporter --web.listen-address=192.168.1.10:9633
```

### Useful flags

From the exporter's README (`smartctl_exporter --help` lists them all):

| Flag | Default | Notes |
|--|--|--|
| `--web.listen-address` | `:9633` | Repeatable. The default listens on every interface; set an address. |
| `--web.telemetry-path` | `/metrics` | The path you paste into Settings. |
| `--smartctl.path` | `/usr/sbin/smartctl` | Where smartmontools put `smartctl`. |
| `--smartctl.interval` | `60s` | How often the exporter polls the disks. ast-context-cache also polls every 60 s. |
| `--smartctl.device` | scan all | Repeatable; monitor only these devices (`/dev/sda`, or with a type such as `/dev/bus/0;megaraid,1`). |
| `--smartctl.device-exclude` / `--smartctl.device-include` | | Regexp filters for the automatic scan (use one or the other). |
| `--smartctl.powermode-check` | `standby` | The power mode check passed to `smartctl`; the default leaves sleeping drives alone instead of spinning them up. |
| `--web.config.file` | | TLS and basic auth for the exporter's own endpoint. |

## Permissions

Run the exporter as **root**. Its README says `smartctl` needs root and that adding the user to the `disk` group is not enough: the kernel always blocks the ATA pass-through commands that SATA SMART reads use for non-root users. The smartmontools answer it quotes notes that the kernel treats a device opened with `CAP_SYS_RAWIO` like one opened by root, so a dedicated user with that capability (for example `AmbientCapabilities=CAP_SYS_RAWIO` in the systemd unit, plus read access to the disk devices through the `disk` group) may work, but the exporter doesn't document that setup and we haven't tested it. The Docker option gets the same access through `privileged: true` and `user: root`.

## 2. Expose it to the guest only

The exporter serves its metrics to anyone who can connect; it has no login unless you configure one. The data isn't secret, but it identifies your hardware down to serial numbers, so don't put it on the internet.

- **Bind to one address.** Use the address the guest reaches the host on: the bridge address (on Proxmox, the `vmbr0` IP of the node), a LAN address, or a Tailscale address if the guest is on your tailnet. Avoid the default `:9633`, which listens on every interface.
- **Firewall the port.** Allow TCP 9633 only from the guest's address, with whatever the host already uses (the Proxmox VE firewall, nftables, ufw, or Tailscale ACLs on a tailnet). Never forward it from your router.
- **Credentials don't go in the URL.** ast-context-cache refuses URLs with `user:password@` and sends no credentials, so the exporter's basic auth (`--web.config.file`) can't be used with it. Restrict by network instead. An `https://` URL works when the exporter's certificate is trusted by the guest's system CA store.

## 3. Verify, then paste the URL

From inside the guest, check that the exporter answers and sees the disks:

```bash
curl -s http://192.168.1.10:9633/metrics | grep smartctl_device
```

You should see one `smartctl_device{...}` line per disk, followed by series such as `smartctl_device_smart_status`, `smartctl_device_temperature` and `smartctl_device_power_on_seconds`. If `curl` hangs or is refused, fix the bind address or firewall first. If it answers but prints nothing, the exporter found no disks it could read: check that it runs as root, and look at its log on the host (`journalctl -u smartctl_exporter`, or `docker logs`).

Then open the dashboard, go to **Settings → Host disks**, paste `http://192.168.1.10:9633/metrics` into **Host disk health URL**, and click **Save**. The server polls the URL right away and then every 60 seconds; the host's disks appear on the Overview disk card with a **Host** chip. Clear the field and save to turn it off.

## What the disk card shows

Each `device` label in the exporter's output becomes one row:

| Card field | Exporter series |
|--|--|
| Device, model, serial, protocol, SMART type | `smartctl_device` labels `device`, `model_name`, `serial_number`, `protocol`, `interface` |
| SMART status (PASSED / FAILED) | `smartctl_device_smart_status` |
| Temperature | `smartctl_device_temperature{temperature_type="current"}` |
| Capacity | `smartctl_device_capacity_bytes` (or `smartctl_device_nvme_capacity_bytes`) |
| Power-on time, power cycles | `smartctl_device_power_on_seconds`, `smartctl_device_power_cycle_count` |
| SSD or HDD | `smartctl_device_rotation_rate` (0 means solid state); NVMe is always solid state |
| Data written | `smartctl_device_bytes_written` |
| NVMe wear, spare, media errors, critical warning | `smartctl_device_percentage_used`, `smartctl_device_available_spare`, `smartctl_device_media_errors`, `smartctl_device_critical_warning` (NVMe disks only) |
| Reallocated, pending, offline uncorrectable, CRC errors | `smartctl_device_attribute{attribute_value_type="raw"}` for `Reallocated_Sector_Ct`, `Current_Pending_Sector`, `Offline_Uncorrectable`, `UDMA_CRC_Error_Count` |
| SAS grown defects, SAS endurance used | `smartctl_scsi_grown_defect_list`, `smartctl_scsi_percentage_used_endurance_indicator` |

A failed SMART check or a non-zero NVMe critical warning marks the disk critical; media errors, reallocated, pending or offline-uncorrectable sectors, and grown defects mark it as a warning. Wear and temperature use the card's usual thresholds.

## When it fails

If a poll fails, the card keeps the last good sample and shows "Can't read host disks" with the reason and the sample's age. The server logs one `WARN` (tag `hostdisk`) when the failure starts or its reason changes, and one `INFO` when it recovers, not one per poll. Changing the URL drops the old sample.

| Card message | Likely cause |
|--|--|
| `fetch host disk health: ... connection refused` or `... deadline exceeded` | The exporter isn't running, listens on a different address, or a firewall drops the guest. Each poll waits at most 5 seconds. |
| `host disk health URL answered 404 Not Found` | Wrong path; the default is `/metrics`. |
| `no smartctl_device metrics in the response` | The URL points at another exporter (node_exporter, for example), or the exporter found no disks it could read (is it running as root?). |
| `parse metrics: ...` | The URL serves something other than Prometheus text, such as an HTML page. |
| `host disk health response is larger than the limit` | The response is over 4 MiB, far more than an exporter sends for a few dozen disks. |
| `refused a redirect to another host or scheme` | The URL redirects elsewhere. Redirects are followed only on the same host, up to three. |

On a container or VM with no URL set, the card says the disks belong to the host and points here. That detection is best-effort (systemd, Docker and Podman container markers, then the CPU's hypervisor flag); setting a URL works whether or not it triggers.
