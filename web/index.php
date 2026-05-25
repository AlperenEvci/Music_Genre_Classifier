<?php
// web/index.php
$maxSizeMB   = 50;
$allowedExts = ['wav', 'mp3'];
?>
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Müzik Türü Sınıflandırma</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/css/style.css" rel="stylesheet">
</head>
<body class="bg-dark text-light">
<div class="container py-5">
    <h1 class="text-center mb-2">Müzik Türü Sınıflandırıcı</h1>
    <p class="text-center text-muted mb-5">Ses dosyanızı yükleyin, tür tahmini ve benzer şarkı önerileri alın.</p>

    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card bg-secondary text-light p-4">
                <form id="uploadForm" action="upload.php" method="POST" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="audioFile" class="form-label">Ses Dosyası (.wav veya .mp3)</label>
                        <input class="form-control" type="file" id="audioFile" name="audioFile"
                               accept=".wav,.mp3" required>
                        <div class="form-text text-muted">Maksimum <?= $maxSizeMB ?> MB</div>
                    </div>
                    <button type="submit" class="btn btn-primary w-100" id="submitBtn">
                        Sınıflandır
                    </button>
                </form>

                <div id="progressSection" class="mt-3 d-none">
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated w-100"
                             role="progressbar">Analiz ediliyor...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="resultSection" class="row justify-content-center mt-4"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="assets/js/app.js"></script>
</body>
</html>
