# Multi-stage build for Go application
FROM golang:1.22-alpine AS builder

# Install git and ca-certificates
RUN apk add --no-cache git ca-certificates

# Set work directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates curl

# Create non-root user
RUN adduser -D -s /bin/sh webapp

# Set work directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Copy static files and templates if they exist
COPY --from=builder /app/static ./static 2>/dev/null || true
COPY --from=builder /app/templates ./templates 2>/dev/null || true

# Change ownership to non-root user
RUN chown -R webapp:webapp /app
USER webapp

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run the application
CMD ["./main"]