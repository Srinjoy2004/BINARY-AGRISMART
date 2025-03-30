// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Navbar scroll behavior
  const navbar = document.getElementById('mainNav');

  window.addEventListener('scroll', function() {
    if (window.scrollY > 50) {
      navbar.classList.add('navbar-shrink');
    } else {
      navbar.classList.remove('navbar-shrink');
    }
  });

  // Back to top button
  const backToTopButton = document.getElementById('backToTop');

  window.addEventListener('scroll', function() {
    if (window.scrollY > 300) {
      backToTopButton.classList.add('active');
    } else {
      backToTopButton.classList.remove('active');
    }
  });

  backToTopButton.addEventListener('click', function(e) {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      if (this.getAttribute('href') !== '#' && !this.getAttribute('href').includes('collapse') && !this.getAttribute('data-bs-toggle')) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          window.scrollTo({
            top: target.offsetTop - 80,
            behavior: 'smooth'
          });
        }
      }
    });
  });

  // Image upload preview
  const fileInput = document.getElementById('disease-image');
  const previewContainer = document.getElementById('image-preview-container');
  const preview = document.getElementById('image-preview');
  const removeButton = document.getElementById('remove-image');

  if (fileInput) {
    fileInput.addEventListener('change', function() {
      const file = this.files[0];
      if (file) {
        const reader = new FileReader();
        reader.addEventListener('load', function() {
          preview.setAttribute('src', this.result);
          previewContainer.classList.remove('d-none');
        });
        reader.readAsDataURL(file);
      }
    });

    removeButton.addEventListener('click', function() {
      preview.setAttribute('src', '#');
      previewContainer.classList.add('d-none');
      fileInput.value = '';
    });
  }

  // Form submissions with simulated responses
  
  document.getElementById("crop-prediction-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent page refresh

    // Collect input values
    const data = {
        N: parseFloat(document.getElementById("nitrogen").value),
        P: parseFloat(document.getElementById("phosphorous").value),
        K: parseFloat(document.getElementById("potassium").value),
        pH: parseFloat(document.getElementById("pH").value),
        rainfall: parseFloat(document.getElementById("rainfall").value),
        temperature: parseFloat(document.getElementById("temperature").value),
        humidity: parseFloat(document.getElementById("humidity").value)
    };

try {
    console.log("Sending request:", data); // Debugging

    // Send request to Flask backend
    const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        // Try to extract error message from response body
        let errorMessage = `HTTP error! Status: ${response.status}`;
        try {
            const errorResponse = await response.json();
            errorMessage += ` - ${errorResponse.message || JSON.stringify(errorResponse)}`;
        } catch (jsonError) {
            console.warn("Could not parse error response JSON");
        }
        throw new Error(errorMessage);
    }

    const result = await response.json();
    console.log("Received response:", result); // Debugging

    let crops = result.top_5_crops || result.recommendations;
    if (crops && crops.length > 0) {
        let dropdown = document.getElementById("crop-dropdown");
        dropdown.innerHTML = ""; // Clear previous options

        crops.forEach(crop => {
            let option = document.createElement("option");
            if (result.top_5_crops) {
                option.value = crop.crop;
                option.textContent = `${crop.crop} (${crop.probability}%)`;
            } else {
                option.value = crop;
                option.textContent = crop;
            }
            dropdown.appendChild(option);
        });

        document.getElementById("crop-selection-section").classList.remove("hidden"); // Show dropdown
        document.getElementById("crop-selection").style.display = "block";
    } else {
        alert("No crops recommended. Try different inputs.");
    }
} catch (error) {
    console.error("Error fetching crop recommendations:", error);
   // alert(error.message);
}
});

document.getElementById("get-fertilizer-btn").addEventListener("click", async function () {
    let selectedCrop = document.getElementById("crop-dropdown").value;

    if (!selectedCrop) {
        alert("Please select a crop first.");
        return;
    }

    try {
        const response = await fetch("/fertilizer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                crop: selectedCrop,
                N: parseFloat(document.getElementById("nitrogen").value),
                P: parseFloat(document.getElementById("phosphorous").value),
                K: parseFloat(document.getElementById("potassium").value)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById("fertilizer-advice").textContent = result.fertilizer_advice;
        document.getElementById("fertilizer-advice-section").classList.remove("hidden"); // Show advice
    } catch (error) {
        console.error("Error fetching fertilizer advice:", error);
        alert("Failed to fetch fertilizer advice. Please check your server.");
    }
});


//image disease prediction
document.getElementById("predict-disease").addEventListener("click", async function (e) {
  e.preventDefault(); // Prevent default form submission

  const fileInput = document.getElementById("disease-image");
  const plantType = document.getElementById("plant-type").value;
  const resultDiv = document.getElementById("disease-result");
  const diseaseName = document.getElementById("disease-name");
  const diseaseDescription = document.getElementById("disease-description");
  const diseaseTreatment = document.getElementById("disease-treatment");
  const submitBtn = document.getElementById("predict-disease");

  if (!fileInput.files.length) {
    alert("Please upload an image first");
    return;
  }

  submitBtn.innerHTML =
    '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Analyzing...';
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);
  formData.append("crop", plantType);

  try {
    const response = await fetch("/predictdisease", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      alert(`Error: ${data.error}`);
    } else {
      diseaseName.textContent = `Disease: ${data.disease}`;
      diseaseDescription.textContent = `Confidence: ${data.confidence}%`;
      diseaseTreatment.textContent = data.treatment || `Detected Disease in ${plantType}`;
      resultDiv.classList.remove("d-none");
    }
  } catch (error) {
    console.error("Prediction error:", error);
    alert("An error occurred while predicting the disease. Please try again.");
  } finally {
    submitBtn.innerHTML = "Analyze Image";
    submitBtn.disabled = false;
    resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});




  // Chatbot functionality
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatbox = document.getElementById('chatbox');

  if (chatForm) {
    chatForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      const message = chatInput.value.trim();
      if (!message) return;
      
      // Add user message to chat
      addMessage(message, 'user');
      chatInput.value = '';
      
      // Simulate bot thinking
      setTimeout(function() {
        // Get bot response
        const response = getBotResponse(message);
        
        // Add bot response to chat
        addMessage(response, 'bot');
        
        // Scroll to bottom
        chatbox.scrollTop = chatbox.scrollHeight;
      }, 1000);
    });
  }

  function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    const messagePara = document.createElement('p');
    messagePara.textContent = text;
    
    const messageTime = document.createElement('div');
    messageTime.classList.add('message-time');
    messageTime.textContent = 'Just now';
    
    messageContent.appendChild(messagePara);
    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);
    
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  function getBotResponse(message) {
    // Simple response system (in a real app, this would use an AI model or predefined responses)
    message = message.toLowerCase();
    
    if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
      return "Hello! How can I assist you with your farming questions today?";
    } else if (message.includes('fertilizer') || message.includes('nutrient')) {
      return "For fertilizer recommendations, I need to know your soil type, current nutrient levels, and what crop you're growing. You can also use our Fertilizer Prediction tool for detailed recommendations.";
    } else if (message.includes('pest') || message.includes('insect')) {
      return "For pest control, it's important to correctly identify the pest first. Could you describe what you're seeing or upload an image to our Disease Detection tool?";
    } else if (message.includes('weather') || message.includes('rain') || message.includes('forecast')) {
      return "I can provide general weather-related farming advice, but for accurate local forecasts, I recommend checking your local weather service or using a weather app specifically designed for agriculture.";
    } else if (message.includes('crop') && message.includes('rotation')) {
      return "Crop rotation is essential for soil health. I recommend rotating between different plant families every season. For example, follow legumes with brassicas, then with alliums, and then with root vegetables.";
    } else if (message.includes('organic') || message.includes('chemical-free')) {
      return "For organic farming, focus on building healthy soil with compost, practicing crop rotation, using beneficial insects for pest control, and applying organic-approved products only when necessary.";
    } else if (message.includes('thank')) {
      return "You're welcome! Feel free to ask if you have any other questions about your farming operation.";
    } else {
      return "That's an interesting question about agriculture. Could you provide more details so I can give you a more specific answer?";
    }
  }

  // Contact form submission
  const contactForm = document.getElementById('contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Simulate loading
      const submitBtn = this.querySelector('button[type="submit"]');
      const originalText = submitBtn.innerHTML;
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Sending...';
      submitBtn.disabled = true;
      
      // Simulate API call with delay
      setTimeout(function() {
        // Create success alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success mt-4';
        alertDiv.role = 'alert';
        alertDiv.innerHTML = '<i class="fas fa-check-circle me-2"></i> Your message has been sent successfully! We\'ll get back to you soon.';
        
        // Add alert to form
        contactForm.appendChild(alertDiv);
        
        // Reset form
        contactForm.reset();
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        // Remove alert after 5 seconds
        setTimeout(function() {
          alertDiv.remove();
        }, 5000);
      }, 1500);
    });
  }

  // Footer newsletter form
  const newsletterForm = document.querySelector('.footer-subscribe-form');
  if (newsletterForm) {
    newsletterForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Simulate API call
      const input = this.querySelector('input[type="email"]');
      const submitBtn = this.querySelector('button[type="submit"]');
      const originalText = submitBtn.innerHTML;
      
      if (!input.value.trim()) return;
      
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>';
      submitBtn.disabled = true;
      
      setTimeout(function() {
        // Create success message
        const successDiv = document.createElement('div');
        successDiv.className = 'alert alert-success mt-3 mb-0';
        successDiv.innerHTML = 'Thanks for subscribing to our newsletter!';
        
        // Replace form with success message
        newsletterForm.innerHTML = '';
        newsletterForm.appendChild(successDiv);
      }, 1000);
    });
  }
});



//price prediction

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("price-prediction").addEventListener("click", function (event) {
      event.preventDefault(); // Prevent form submission

      let state = document.getElementById("state").value;
      let district = document.getElementById("district").value;
      let crop = document.getElementById("crop").value;
      let season = document.getElementById("season").value;
      let area = document.getElementById("area").value;
      let year = document.getElementById("year").value;

      if (!state || !district || !crop || !season || !area || !year) {
          alert("Please fill in all fields.");
          return;
      }

      let requestData = {
          state: state,
          district: district,
          crop: crop,
          season: season,
          area: area,
          year: year
      };

      fetch("/predict-price", {
          method: "POST",
          headers: {
              "Content-Type": "application/json"
          },
          body: JSON.stringify(requestData)
      })
      .then(response => response.json())
      .then(data => {
          if (data.error) {
              alert("Error: " + data.error);
          } else {
              document.getElementById("production-advice").innerHTML = `<strong>${data.prediction}</strong>`;
          }
      })
      .catch(error => console.error("Error:", error));
  });
});
