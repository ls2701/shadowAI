"""
generate_dummy_logs.py

Produces ~650 synthetic log rows in the format:
    timestamp,sensor,source,event

Covers the log formats your Shadow AI pipeline understands:
  - Sophos firewall KV       (device_name="SFW" ...)
  - SonicWall firewall KV    (id=firewall sn=... ...)
  - Arctic Wolf agent JSON   (@type="awn-agent"...)
  - Windows DNS packet log   (MSWinEventLog ... PACKET ...)
  - Windows Event Log        (MSWinEventLog ... System/Security ...)
  - AWS CloudTrail JSON
  - Office 365 sign-in JSON

Includes a mix of:
  * Real AI product domains  (chatgpt.com, claude.ai, gemini.google.com, ...)
  * Plausible fake AI hosts  (llm-gateway.azure.com, ai-copilot.corp.internal, ...)
  * Non-AI noise             (microsoft.com, arcticwolf.net, youtube.com, ...)
  * Plenty of single-shot domains so the classifier is exercised
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(1337)

N_LOGS = 650
OUT_FILE = Path("dummy_logs.csv")

START = datetime(2026, 8, 2, 23, 0, 0)

# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------
SENSORS = {
    "sophos":       ("Acme India (HO)",   "192.168.0.1"),
    "sonicwall":    ("US Sensor",         "192.168.101.1"),
    "arcticwolf":   ("acme-workstations", "agent"),
    "windows_dns":  ("US Sensor",         "acme-dns"),
    "mswineventlog":("US Sensor",         "acme-mswineventlog"),
    "cloudtrail":   ("acme-aws",          "acme-org-aws-cloudtrail-logs-111122223333"),
    "o365":         ("acme-tenant",       "office 365"),
}

USERS = [
    "AliceSmith", "BobJones", "CarolWilliams", "DavidBrown",
    "EveJohnson", "FrankGarcia", "GraceMiller", "HenryDavis",
    "IvyRodriguez", "JackMartinez", "KateWilson", "LeoAnderson",
    "MiaThomas", "NoahTaylor", "OliviaMoore", "PeterJackson",
    "QuinnHarris", "RachelClark", "SamLewis", "TinaWalker",
]

INTERNAL_IPS = [f"192.168.101.{n}" for n in (9, 21, 54, 55, 58, 61, 66, 69,
                                             86, 87, 160, 196, 198, 199, 200)]
INTERNAL_SUBNET = [f"192.168.0.{n}" for n in (7, 8, 9, 11, 18, 19, 22, 33)]

# Real AI products – some should match your static list, some should not
AI_REAL_DOMAINS = [
    "chatgpt.com",
    "api.openai.com",
    "claude.ai",
    "api.anthropic.com",
    "gemini.google.com",
    "perplexity.ai",
    "cursor.com",
    "huggingface.co",
    "monica.im",
    "sider.ai",
    "getmerlin.in",
]

# Plausible AI-ish hosts (fake) to exercise the dynamic classifier
AI_FAKE_DOMAINS = [
    "chat.mistral.ai",
    "api.cohere.ai",
    "replicate.com",
    "runpod.ai",
    "modal.com",
    "togetherai.com",
    "copy.ai",
    "jasper.ai",
    "midjourney.com",
    "stability.ai",
    "writer.com",
    "pi.ai",
    "character.ai",
    "poe.com",
    "llm-gateway.azure.com",
    "ai-copilot.corp.internal",
    "inference.deepmind.example",
    "ai-assistant.acme.com",
    "prompt-hub.acme.com",
    "genai-platform.acme.io",
]

# Non-AI noise
NON_AI_DOMAINS = [
    "microsoft.com",
    "outlook.office365.com",
    "login.microsoftonline.com",
    "sharepoint.com",
    "arcticwolf.net",
    "vipre.com",
    "youtube.com",
    "google.com",
    "amazonaws.com",
    "github.com",
    "atlassian.com",
    "zoom.us",
    "slack.com",
    "salesforce.com",
    "aws.amazon.com",
    "wpad.acme.local",
    "teredo.ipv6.microsoft.com",
    "ctldl.windowsupdate.com",
    "settings-win.data.microsoft.com",
    "time.windows.com",
]

AWS_SERVICES = [
    "sts.amazonaws.com",
    "s3.amazonaws.com",
    "ec2.amazonaws.com",
    "elasticloadbalancing.amazonaws.com",
    "kms.amazonaws.com",
    "config.amazonaws.com",
    "securityhub.amazonaws.com",
    "tagging.amazonaws.com",
    "bedrock.amazonaws.com",       # AI hint
    "sagemaker.amazonaws.com",     # AI hint
]

O365_APPS = [
    ("Office 365 Exchange Online", "00000002-0000-0ff1-ce00-000000000000"),
    ("Microsoft Teams",            "1fec8e78-bce4-4aaf-ab1b-5451cc387264"),
    ("SharePoint Online",          "00000003-0000-0ff1-ce00-000000000000"),
    ("Microsoft Graph",            "00000003-0000-0000-c000-000000000000"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ts(offset_seconds: float) -> str:
    return (START + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def pick_user() -> str:
    return random.choice(USERS)


def pick_domain() -> str:
    """Weighted: mostly real/fake AI domains so we get plenty of hits."""
    r = random.random()
    if r < 0.55:
        return random.choice(AI_REAL_DOMAINS + AI_FAKE_DOMAINS)
    return random.choice(NON_AI_DOMAINS)


# ---------------------------------------------------------------------------
# Log generators
# ---------------------------------------------------------------------------
def sophos_row(off: float) -> tuple[str, str, str, str]:
    sensor, ip = SENSORS["sophos"]
    domain = pick_domain()
    user = pick_user()
    src = random.choice(INTERNAL_SUBNET)
    bytes_sent = random.randint(200, 8000)
    bytes_recv = random.randint(200, 12000)
    ts_local = (START + timedelta(seconds=off)).strftime("%Y-%m-%dT%H:%M:%S+0530")
    event = (
        f'device_name="SFW" timestamp="{ts_local}" device_model="XGS2100" '
        f'device_serial_id="X21010389TJ3H54" log_id="010101600001" '
        f'log_type="Firewall" log_component="Firewall Rule" '
        f'log_subtype="Allowed" log_version=1 severity="Information" '
        f'duration={random.randint(1, 60)} fw_rule_id="1" fw_rule_name="Internet" '
        f'fw_rule_section="Local rule" nat_rule_id="1" nat_rule_name="internet" '
        f'fw_rule_type="USER" gw_id_request=2 gw_name_request="ACT 500Mbps" '
        f'web_policy_id=13 ips_policy_id=8 app_filter_policy_id=1 '
        f'ether_type="Unknown (0x0000)" in_interface="Port1" out_interface="Port3" '
        f'src_mac="00:90:0B:E9:E6:FD" dst_mac="7C:5A:1C:BE:97:CA" '
        f'src_ip="{src}" src_country="R1" dst_ip="{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" '
        f'dst_country="USA" protocol="TCP" src_port={random.randint(10000,65535)} '
        f'dst_port=443 packets_sent={random.randint(1,30)} packets_received={random.randint(1,30)} '
        f'bytes_sent={bytes_sent} bytes_received={bytes_recv} '
        f'src_trans_ip="106.51.77.128" src_zone_type="LAN" src_zone="LAN" '
        f'dst_zone_type="WAN" dst_zone="WAN" con_event="Stop" '
        f'con_id={random.randint(100000000, 4000000000)} hb_status="No Heartbeat" '
        f'app_resolved_by="Signature" app_is_cloud="FALSE" qualifier="New" '
        f'in_display_interface="Port1" out_display_interface="Port3" '
        f'log_occurrence="1" user="{user}" dstname="{domain}"'
    )
    return ts(off), sensor, ip, event


def sonicwall_row(off: float) -> tuple[str, str, str, str]:
    sensor, ip = SENSORS["sonicwall"]
    domain = pick_domain()
    user = pick_user()
    src = random.choice(INTERNAL_IPS)
    ts_local = (START + timedelta(seconds=off)).strftime("%Y-%m-%d %H:%M:%S")
    event = (
        f'id=firewall sn=18C241CBC520 time="{ts_local}" fw=70.89.42.137 '
        f'pri=6 c=1024 gcat=2 m=97 msg="Web site hit" '
        f'srcMac=cc:48:3a:90:13:b4 src={src}:59891:X0 srcZone=LAN '
        f'natSrc=70.89.42.137:64260 dstMac=3c:2d:9e:d2:60:63 '
        f'dst={random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}:443:X1 dstZone=WAN '
        f'natDst=18.224.11.24:443 proto=tcp/https '
        f'sent={random.randint(100, 8000)} rcvd={random.randint(100, 9000)} '
        f'rule="Default Access Rule" app=11 dstname={domain} arg=/ code=76 '
        f'Category="Computer and Internet Security" '
        f'note="Policy: cfsZonePolicy0, Info: 5892" '
        f'n={random.randint(8000000, 9000000)} fw_action="forward" dpi=0 '
        f'usr="{user}"'
    )
    return ts(off), sensor, ip, event


def arcticwolf_row(off: float) -> tuple[str, str, str, str]:
    sensor, source = SENSORS["arcticwolf"]
    user = pick_user()
    hostname = f"{random.randint(1000,9999)}{random.choice(['HC','US','IN','UK'])}{user.upper()[:8]}"
    src_ip = random.choice(INTERNAL_IPS)
    eid = random.choice([200007, 200013, 164662, 164673, 164688])
    if eid == 200007:
        desc = "Sysmon - Image Loaded"
        inner_extra = '"imageLoaded":"C:\\\\Windows\\\\System32\\\\wbem\\\\wmiutils.dll"'
    elif eid == 200013:
        desc = "Sysmon - RegValue Set"
        inner_extra = '"targetObject":"HKU\\\\S-1-5-21-0000000000\\\\Software\\\\Microsoft\\\\Office\\\\Outlook\\\\Addins\\\\MAPI_add_in.Connect\\\\LoadBehavior"'
    elif eid == 164662:
        desc = "An operation was performed on an object"
        inner_extra = '"objectServer":"WMI","objectType":"WMI Namespace"'
    elif eid == 164673:
        desc = "A privileged service was called"
        inner_extra = '"service":"LsaRegisterLogonProcess()","privilegeList":"SeTcbPrivilege"'
    else:
        desc = "A new process has been created"
        inner_extra = '"newProcessName":"C:\\\\Windows\\\\System32\\\\conhost.exe"'

    payload = {
        "@type": "awn-agent",
        "tenant": {
            "deployment_id": "acme",
            "customer_uuid": "aaaa1111-bbbb-2222-cccc-333344445555",
        },
        "host": {
            "interface_name": "Ethernet",
            "ip": src_ip,
            "public_ip": "203.0.113.42",
            "mac": "0a:00:27:00:00:1a",
            "hostname": hostname,
            "user": {"name": user},
            "os": {"platform": "windows", "name": "Microsoft Windows 11"},
        },
        "event": {
            "id": str(eid),
            "timestamp": ts(off),
            "risk_score": 3,
            "risk_score_norm": 20,
            "description": desc,
            "module": "eventchannel",
            "original": "{" + inner_extra + "}",
        },
        "agent": {
            "id": "11111111-2222-3333-4444-555566667777",
            "server_name": "worker-7",
        },
    }
    return ts(off), sensor, source, json.dumps(payload, separators=(",", ":"))


def windows_dns_row(off: float) -> tuple[str, str, str, str]:
    sensor, source = SENSORS["windows_dns"]
    domain = pick_domain()
    labels = domain.split(".")
    encoded = "".join(f"({len(l)}){l}" for l in labels) + "(0)"
    client = random.choice(INTERNAL_IPS)
    ts_local = (START + timedelta(seconds=off)).strftime("%m/%d/%Y %I:%M:%S %p")
    event = (
        f'MSWinEventLog\t1\tN/A\t{random.randint(8000000, 9000000)}\t'
        f'{ts_local}\tN/A\tN/A\tN/A\tN/A\tN/A\tACME-DNS\tN/A\t\t'
        f'{ts_local} 1410 PACKET  0000000002844070 UDP Rcv {client} be12   '
        f'Q [0001   D   NOERROR] A      {encoded}\tN/A'
    )
    return ts(off), sensor, source, event


def mswineventlog_row(off: float) -> tuple[str, str, str, str]:
    sensor, source = SENSORS["mswineventlog"]
    hostname = "ACME-MSWIN"
    ev_time = (START + timedelta(seconds=off)).strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "EventTime": ev_time,
        "Hostname": hostname,
        "EventType": "ERROR",
        "SeverityValue": 4,
        "Severity": "ERROR",
        "EventID": 36887,
        "SourceName": "Schannel",
        "Channel": "System",
        "Domain": "NT AUTHORITY",
        "AccountName": "SYSTEM",
        "UserID": "S-1-5-18",
        "Message": "The following fatal alert was received: 70.",
        "AlertDesc": "70",
        "EventReceivedTime": ev_time,
        "SourceModuleName": "in_EVENT",
        "SourceModuleType": "im_msvistalog",
    }
    event = (
        f'MSWinEventLog\t3\tSystem\t{random.randint(8000000, 9000000)}\t'
        f'{ev_time}\t36887\tSchannel\tSYSTEM\tUser\tError\t{hostname}\tN/A\t\t'
        f'{json.dumps(payload, separators=(",", ":"))}\t{random.randint(8000000, 9000000)}'
    )
    return ts(off), sensor, source, event


def cloudtrail_row(off: float) -> tuple[str, str, str, str]:
    sensor, source = SENSORS["cloudtrail"]
    svc = random.choice(AWS_SERVICES)
    event_name = {
        "sts.amazonaws.com": "AssumeRole",
        "s3.amazonaws.com": "HeadObject",
        "ec2.amazonaws.com": "DescribeNetworkInterfaces",
        "elasticloadbalancing.amazonaws.com": "DescribeTargetHealth",
        "kms.amazonaws.com": "GenerateDataKey",
        "config.amazonaws.com": "PutEvaluations",
        "securityhub.amazonaws.com": "GetResources",
        "tagging.amazonaws.com": "GetResources",
        "bedrock.amazonaws.com": "InvokeModel",
        "sagemaker.amazonaws.com": "CreateEndpoint",
    }[svc]

    payload = {
        "eventVersion": "1.11",
        "userIdentity": {
            "type": "AssumedRole",
            "principalId": "AROAEXAMPLE:session",
            "arn": "arn:aws:sts::111122223333:assumed-role/AcmeRole/session",
            "accountId": "111122223333",
            "userName": pick_user(),
        },
        "eventTime": ts(off).replace("+00:00", "Z"),
        "eventSource": svc,
        "eventName": event_name,
        "awsRegion": "us-east-1",
        "sourceIPAddress": "203.0.113.42",
        "userAgent": "Boto3/1.38.46",
        "requestParameters": {"roleArn": "arn:aws:iam::111122223333:role/AcmeRole"},
        "responseElements": None,
        "requestID": "00000000-0000-0000-0000-000000000001",
        "eventID": "00000000-0000-0000-0000-000000000002",
        "readOnly": True,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": "111122223333",
    }
    return ts(off), sensor, source, json.dumps(payload, separators=(",", ":"))


def o365_row(off: float) -> tuple[str, str, str, str]:
    sensor, source = SENSORS["o365"]
    user = pick_user()
    app_name, app_id = random.choice(O365_APPS)
    payload = {
        "id": "00000000-0000-0000-0000-00000000000A",
        "createdDateTime": ts(off).replace("+00:00", "Z"),
        "userDisplayName": user,
        "userPrincipalName": f"{user.lower()}@acme.example",
        "userId": "00000000-0000-0000-0000-00000000000B",
        "appId": app_id,
        "appDisplayName": app_name,
        "ipAddress": "203.0.113.42",
        "clientAppUsed": "Browser",
        "userAgent": "Mozilla/5.0",
        "correlationId": "00000000-0000-0000-0000-00000000000C",
        "conditionalAccessStatus": "success",
        "isInteractive": True,
        "resourceDisplayName": app_name,
        "homeTenantId": "aaaa1111-bbbb-2222-cccc-333344445555",
        "userType": "member",
        "deviceDetail": {
            "displayName": "ACME-LAPTOP",
            "operatingSystem": "Windows",
            "browser": "Edge",
            "isCompliant": True,
            "isManaged": True,
            "trustType": "Azure AD joined",
        },
        "location": {
            "city": "Bengaluru",
            "state": "Karnataka",
            "countryOrRegion": "IN",
        },
        "status": {"errorCode": 0, "failureReason": "Success."},
    }
    return ts(off), sensor, source, json.dumps(payload, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
GENERATORS = [
    (sophos_row,        0.26),
    (sonicwall_row,     0.20),
    (arcticwolf_row,    0.18),
    (windows_dns_row,   0.15),
    (mswineventlog_row, 0.05),
    (cloudtrail_row,    0.10),
    (o365_row,          0.06),
]


def pick_generator():
    r = random.random()
    acc = 0.0
    for gen, weight in GENERATORS:
        acc += weight
        if r <= acc:
            return gen
    return GENERATORS[-1][0]


def main():
    rows = []
    for i in range(N_LOGS):
        off = i * random.uniform(15, 45)   # 15–45s apart, ~6h span
        gen = pick_generator()
        rows.append(gen(off))

    rows.sort(key=lambda r: r[0])   # chronological

    with OUT_FILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(["timestamp", "sensor", "source", "event"])
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()