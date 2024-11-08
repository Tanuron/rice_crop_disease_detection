document.getElementById('imageInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.style.backgroundImage = `url(${e.target.result})`;
            imagePreview.style.display = 'block';  // Show the preview box
            document.getElementById('submitBtn').style.display = 'inline-block';  // Show the submit button
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('submitBtn').addEventListener('click', function () {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        console.log('Sending request to server...');
        
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Received response from server...');
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            const outputDiseaseName = document.getElementById('outputDiseaseName');
            if (data.error) {
                outputDiseaseName.innerText = `Error: ${data.error}`;
            } else {
                outputDiseaseName.innerText = `Predicted Disease: ${data.prediction}\nConfidence: ${data.confidence.toFixed(2)}%`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            const outputDiseaseName = document.getElementById('outputDiseaseName');
            outputDiseaseName.innerText = `Error: ${error.message}`;
        });
    } else {
        alert('Please select an image first.');
    }
});
