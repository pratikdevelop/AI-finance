# Use Node.js 18 based on Alpine for a smaller image
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (will be created later)
COPY package*.json ./

# Install dependencies inside the container
RUN npm install

# Install Angular CLI globally
RUN npm install -g @angular/cli



# Copy the rest of the application code
COPY . .

# Expose port 4200 for Angular development server
EXPOSE 4200

# Command to start the Angular development server
CMD ["npm", "start"]
