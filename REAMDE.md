# Neighbor API (Local)

Local FastAPI service that runs your optimization code.  
Send JSON like `[{"length": 10, "quantity": 1}]`, get back a list of
`{location_id, listing_ids, total_price_in_cents}` sorted by price.

---

## Requirements

- `listings.json` in the same folder as `Neighbor.py`
- Virtual environment (recommended)

Dependencies are listed in `requirements.txt`.

---


### Windows (PowerShell)
```powershell
cd C:\path\to\Neighbor
python -m venv venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   # once
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
