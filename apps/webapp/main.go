package main

import (
	"log"

	"chemplusx.com/midas/controller"
	"chemplusx.com/midas/server"
	"chemplusx.com/midas/service"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "pong",
		})
	})
	log.Println("Starting server")
	config := service.DefaultConfig()
	// Customize config if needed

	controller.WS = service.NewService(config)
	if err := controller.WS.Start(); err != nil {
		log.Fatal(err)
	}
	defer controller.WS.Stop()
	log.Println("Server started")
	server.InitRouter(r)
	r.Run(":5050")
}
