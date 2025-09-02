#!/bin/bash
# Set up Google Cloud secrets for Pitch Perfect deployment

PROJECT_ID="pitchperfect-lewagon"

echo "🔐 Setting up Google Cloud Secrets for Pitch Perfect"
echo "=================================================="
echo ""

# Set the project
gcloud config set project $PROJECT_ID

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local secret_description=$2
    
    echo ""
    echo "Setting up secret: $secret_name"
    
    # Check if secret exists
    if gcloud secrets describe $secret_name &> /dev/null; then
        echo "Secret already exists. Do you want to update it? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Please enter the new value for $secret_description:"
            read -s secret_value
            echo -n "$secret_value" | gcloud secrets versions add $secret_name --data-file=-
            echo "✅ Secret $secret_name updated"
        else
            echo "⏭️  Skipping $secret_name"
        fi
    else
        echo "Creating new secret: $secret_name"
        echo "Please enter the value for $secret_description:"
        read -s secret_value
        echo -n "$secret_value" | gcloud secrets create $secret_name --data-file=-
        echo "✅ Secret $secret_name created"
    fi
}

# Create/update secrets
create_or_update_secret "openai-api-key" "OpenAI API Key"
create_or_update_secret "elevenlabs-api-key" "ElevenLabs API Key"

echo ""
echo "✅ Secret setup complete!"
echo ""
echo "You can now run the GitHub Actions workflow for deployment."