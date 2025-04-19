
const textArea = document.getElementById('text-area');
const predictionsContainer = document.getElementById('predictions');

textArea.addEventListener('input', async function(event) {
    const text = textArea.value.trim();

    if (text === '') {
        predictionsContainer.innerHTML = '';
        return;
    }
    if (event.inputType === 'insertText' && event.data === ' ') {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (response.ok) {
            const data = await response.json();
            displayPredictions(data.predictions);
        }
    }
});

function displayPredictions(predictions) {
    predictionsContainer.innerHTML = '';

    predictions.forEach(word => {
        const wordElement = document.createElement('span');
        wordElement.classList.add('prediction-box');
        wordElement.textContent = word;

        wordElement.onclick = function() {
            textArea.value = textArea.value.trimEnd() + ' ' + word + ' ';
            predictionsContainer.innerHTML = '';
        };

        predictionsContainer.appendChild(wordElement);
    });
}
