import urllib.request
import json
import zipfile
import os

print("Fetching latest Quantum ESPRESSO Windows release info from QMatSuite...")
url = "https://api.github.com/repos/QMatSuite/quantum-espresso-windows-exe/releases/latest"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        
    dl_url = None
    for asset in data.get('assets', []):
        if asset['name'].endswith('.zip'):
            dl_url = asset['browser_download_url']
            break
            
    if not dl_url:
        print("Could not find a .zip asset in the latest release.")
        exit(1)
        
    print(f"Downloading from: {dl_url}")
    zip_path = "qe_win.zip"
    
    # Simple chunked download to avoid memory issues and show progress
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        # Re-requesting the actual asset URL now
        req_dl = urllib.request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_dl) as dl_response:
            shutil.copyfileobj(dl_response, out_file)
            
except Exception as e:
    import urllib.request as request
    # Retry with direct urllib request
    req = request.Request(dl_url, headers={'User-Agent': 'Mozilla/5.0'})
    with request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())

print("Extracting...")
extract_dir = "qe_binaries"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

os.remove(zip_path)
print(f"Installation complete. Binaries extracted to {os.path.abspath(extract_dir)}")
