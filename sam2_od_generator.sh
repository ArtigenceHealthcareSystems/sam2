check_command() {
    if [ $? -ne 0 ]; then
        echo "$1 failed. Exiting."
        exit 1
    fi
}

get_all_versions() {
    local registry_name="$1"
    local image_name="$2"
    
    echo "Fetching all versions from $registry_name/$image_name..."
    local versions=$(az acr repository show-tags --name "$registry_name" --repository "$image_name" --orderby time_desc -o tsv)
    
    if [ -z "$versions" ]; then
        echo "No versions found in the repository."
        return
    fi
    
    echo "Available versions:"
    echo "$versions" | while read -r version; do
        echo "  - $version"
    done
}

get_current_version() {
    local registry_name="$1"
    local image_name="$2"
    
    echo "Fetching current version from $registry_name/$image_name..."
    local current_version=$(az acr repository show-tags --name "$registry_name" --repository "$image_name" --orderby time_desc --query "[0]" -o tsv)
    
    if [ -z "$current_version" ]; then
        echo "No previous version found. Starting with version v1.0"
        echo "v1.0"
    else
        echo "Current version: $current_version"
        echo "$current_version"
    fi
}

validate_version() {
    local version="$1"
    # Accept formats like v1.9, v1.0, or just 1.9, 1.0
    if [[ ! $version =~ ^v?[0-9]+\.[0-9]+$ ]]; then
        return 1
    fi
    return 0
}

# Remove 'v' prefix for Docker tags
get_docker_tag() {
    local version="$1"
    echo "${version#v}"
}

push_image_to_acr() {
    local registry_name="$1"
    local image_name="$2"
    local dockerfile="$3"
    local version="$4"
    local docker_tag=$(get_docker_tag "$version")

    echo "Building Docker image: $image_name using $dockerfile..."
    docker build -t "$image_name" -f "$dockerfile" .
    check_command "Docker build for $image_name"

    echo "Tagging Docker image: $image_name as $registry_name/$image_name:$docker_tag"
    docker tag "$image_name" "$registry_name/$image_name:$docker_tag"
    check_command "Tagging Docker image"

    echo "Tagging Docker image: $image_name as $registry_name/$image_name:latest"
    docker tag "$image_name" "$registry_name/$image_name:latest"
    check_command "Tagging Docker image"

    echo "Logging in to Azure Container Registry: $registry_name"
    az acr login --name "$registry_name"
    check_command "Azure Container Registry login"

    echo "Pushing Docker image to $registry_name/$image_name:$docker_tag..."
    docker push "$registry_name/$image_name:$docker_tag"
    check_command "Pushing Docker image"

    echo "Pushing latest tag to $registry_name/$image_name:latest..."
    docker push "$registry_name/$image_name:latest"
    check_command "Pushing latest tag"
}

main() {
    local registry_name="blooddrop.azurecr.io"
    local image_name="sam2_od"
    local dockerfile="sam2_od_Dockerfile"
    
    # Display all versions
    get_all_versions "$registry_name" "$image_name"
    
    # Get current version
    current_version=$(get_current_version "$registry_name" "$image_name")
    
    # Ask user for new version
    echo -e "\nCurrent version: $current_version"
    echo "Enter new version number (format: vX.Y or X.Y, e.g., v1.9 or 1.9):"
    read -p "New version: " new_version
    
    # Validate version format
    while ! validate_version "$new_version"; do
        echo "Invalid version format. Please use the format vX.Y or X.Y (e.g., v1.9 or 1.9)"
        read -p "New version: " new_version
    done
    
    echo "Using version: $new_version"
    push_image_to_acr "$registry_name" "$image_name" "$dockerfile" "$new_version"
    echo "All operations completed successfully."
}

main 