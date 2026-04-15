const labelsIndo = ["Bangunan", "Hutan", "Gletser", "Pegunungan", "Lautan", "Jalanan"];

async function runInference(imgElement) {
    const panel = document.getElementById('analysisPanel');
    const text = document.getElementById('predictionText');
    
    panel.style.display = 'block';
    text.innerText = 'Menganalisis citra...';

    try {
        const model = await tf.loadGraphModel('./public/model.json');
        const tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([150, 150])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

        const predictions = await model.predict(tensor).data();
        const maxIdx = predictions.indexOf(Math.max(...predictions));
        
        // Output formal: Menampilkan hasil akhir
        text.innerText = labelsIndo[maxIdx];
    } catch (err) {
        text.innerText = 'Gagal memproses analisis.';
    }
}

document.getElementById('imageSelector').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('uploadStatus').innerText = file.name;
        const reader = new FileReader();
        reader.onload = () => {
            const preview = document.getElementById('preview');
            preview.src = reader.result;
            document.getElementById('previewWrapper').style.display = 'block';
            preview.onload = () => runInference(preview);
        };
        reader.readAsDataURL(file);
    }
});