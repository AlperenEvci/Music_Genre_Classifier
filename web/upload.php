<?php
// web/upload.php
header('Content-Type: application/json; charset=utf-8');
set_time_limit(120);

$allowedExts  = ['wav', 'mp3'];
$maxSizeBytes = 10 * 1024 * 1024;
$uploadDir    = __DIR__ . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . 'tmp' . DIRECTORY_SEPARATOR;

// Python binary — Windows path, gerekirse düzenle
// Eğer virtualenv varsa: proje_root/venv/Scripts/python.exe
$pythonBin    = realpath(__DIR__ . '/../venv/Scripts/python.exe');
if (!$pythonBin) {
    $pythonBin = 'python'; // PATH'te python varsa
}
$predictScript = realpath(__DIR__ . '/../scripts/predict.py');

function json_error(string $msg, int $code = 400): void {
    http_response_code($code);
    echo json_encode(['error' => $msg, 'genre' => null, 'confidence' => 0.0, 'recommendations' => []]);
    exit;
}

if (!isset($_FILES['audioFile']) || $_FILES['audioFile']['error'] !== UPLOAD_ERR_OK) {
    json_error('Dosya yüklenemedi.', 400);
}

$file = $_FILES['audioFile'];
$ext  = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));

if (!in_array($ext, $allowedExts, true)) {
    json_error('Geçersiz dosya uzantısı. Yalnızca .wav veya .mp3 kabul edilir.', 422);
}

if ($file['size'] > $maxSizeBytes) {
    json_error('Dosya çok büyük. Maksimum 10 MB.', 422);
}

// MIME type ek kontrol — hard block
$mimeType = mime_content_type($file['tmp_name']);
$allowedMimes = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'];
if (!in_array($mimeType, $allowedMimes, true)) {
    @unlink($tmpFile);
    json_error('Geçersiz dosya türü.', 422);
}

$tmpFile = $uploadDir . bin2hex(random_bytes(8)) . '.' . $ext;

if (!move_uploaded_file($file['tmp_name'], $tmpFile)) {
    json_error('Dosya kaydedilemedi.', 500);
}

if (!$predictScript) {
    @unlink($tmpFile);
    json_error('predict.py scripti bulunamadı.', 500);
}

/*
 * PLAN C — Flask API (shell_exec çalışmazsa bu bloğu kullan, aşağıdaki shell_exec bloğunu kaldır)
 *
 * $ch = curl_init('http://127.0.0.1:5000/predict');
 * curl_setopt_array($ch, [
 *     CURLOPT_POST           => true,
 *     CURLOPT_POSTFIELDS     => ['audio' => new CURLFile($tmpFile)],
 *     CURLOPT_RETURNTRANSFER => true,
 *     CURLOPT_TIMEOUT        => 60,
 * ]);
 * $output = curl_exec($ch);
 * $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
 * curl_close($ch);
 * @unlink($tmpFile);
 * if ($httpCode !== 200) {
 *     json_error('Flask API hatası', 500);
 * }
 * $data = json_decode($output, true);
 * echo json_encode($data, JSON_UNESCAPED_UNICODE);
 * exit;
 */

$cmd    = escapeshellarg($pythonBin) . ' ' . escapeshellarg($predictScript) . ' ' . escapeshellarg($tmpFile) . ' 2>&1';
$output = shell_exec($cmd);

@unlink($tmpFile);

if ($output === null) {
    json_error('Python scripti çalıştırılamadı. shell_exec devre dışı olabilir.', 500);
}

$output = trim($output);
$data   = json_decode($output, true);

if ($data === null) {
    error_log('Python parse error: ' . $output);
    json_error('Ses dosyası işlenirken bir hata oluştu.', 500);
}

echo json_encode($data, JSON_UNESCAPED_UNICODE);
