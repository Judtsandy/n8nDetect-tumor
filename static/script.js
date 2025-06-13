function predict() {
    const fileInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const previewImg = document.getElementById('imagePreview');
    const tumorPreview = document.getElementById('tumorPreview'); // Nueva referencia

    if (!fileInput.files[0]) {
        resultDiv.innerHTML = "Por favor, selecciona una imagen.";
        return;
    }

    // Vista previa de la imagen antes de enviar
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
    };
    reader.readAsDataURL(fileInput.files[0]);

    // Enviar la imagen al backend
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    resultDiv.innerHTML = "Analizando...";
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        } else {
            // Mostrar resultado con color condicional
            resultDiv.innerHTML = `${data.result} (Confianza: ${data.confidence.toFixed(2)}%)`;
            resultDiv.style.color = data.result.includes('Tumor') ? '#d81b60' : '#00796b';

            // Actualizar imagen original con la devuelta por el backend
            previewImg.src = data.img_original;
            previewImg.style.display = 'block';

            // Mostrar imagen con tumor resaltado si existe
            if (data.img_tumor) {
                tumorPreview.src = data.img_tumor;
                tumorPreview.style.display = 'block';
            } else {
                tumorPreview.src = '';
                tumorPreview.style.display = 'none';
            }
        }
    })
    .catch(error => {
        resultDiv.innerHTML = "Error al analizar la imagen.";
        console.error(error);
    });
}
