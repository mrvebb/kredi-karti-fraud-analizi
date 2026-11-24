# Öğrenci 1 -> Öğrenci 3 -> Öğrenci 2 akışını sırayla çalıştırır.
# Her script kesintiye dayanıklı: tekrar çalıştırınca bitmiş adımları atlar, kaldığı yerden devam eder.
#
# VS Code'dan BAĞIMSIZ (pencere kapansa da devam eder) başlatmak için, herhangi bir PowerShell'de:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$PWD\run_all.ps1" -WindowStyle Minimized
# İlerlemeyi izlemek:
#   Get-Content results\run_all.log -Wait -Tail 30
#
# Parametreler:  .\run_all.ps1 -Steps 2 -Sizes 5000000 -DriverMemory 8g
param(
    [int[]]$Steps = @(1, 3, 2),
    [int[]]$Sizes = @(100000, 1000000, 5000000),
    [string]$DriverMemory = "8g"
)
# "Continue": PS 5.1'de python'un stderr satırları (Spark uyarıları) hata sayılıp betiği durdurmasın;
# başarı/başarısızlık python'un çıkış koduyla ($LASTEXITCODE) belirlenir.
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$log = Join-Path $PSScriptRoot "results\run_all.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

# Dosyaya yönlendirmede Türkçe karakterler bozulmasın
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

function Write-Log([string]$msg) {
    $line = "{0} | run_all | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

$sizeArgs = $Sizes | ForEach-Object { "$_" }
$commands = @{
    1 = @("src\ogrenci1_veri_hazirlama.py", "--sizes") + $sizeArgs + @("--driver_memory", $DriverMemory)
    3 = @("src\ogrenci3_ozellik_uretimi.py", "--sizes") + $sizeArgs + @("--driver_memory", $DriverMemory)
    2 = @("src\ogrenci2_class_imbalance.py", "--sizes") + $sizeArgs + @("--driver_memory", $DriverMemory)
}

Write-Log "BAŞLADI  adımlar=$($Steps -join ',') boyutlar=$($Sizes -join ',') pid=$PID"
foreach ($s in $Steps) {
    if (-not $commands.ContainsKey($s)) { Write-Log "Öğrenci $s bu betikte yok, atlandı"; continue }
    Write-Log "Öğrenci $s BAŞLADI: python $($commands[$s] -join ' ')"
    $t0 = Get-Date
    # python çıktısı (Spark uyarıları dahil) aynı log dosyasına eklenir
    # (Tee-Object PS 5.1'de UTF-16 yazdığı için satır satır UTF-8 ekleniyor)
    & $py -u @($commands[$s]) 2>&1 | ForEach-Object {
        $l = "$_"
        Write-Host $l
        Add-Content -Path $log -Value $l -Encoding utf8
    }
    $code = $LASTEXITCODE
    $dk = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) {
        Write-Log "Öğrenci $s HATA (çıkış kodu $code, $dk dk). Düzeltip aynı komutla tekrar başlatın; bitmiş adımlar atlanır."
        exit $code
    }
    Write-Log "Öğrenci $s BİTTİ ($dk dk)"
}
Write-Log "TÜM ADIMLAR TAMAMLANDI"
