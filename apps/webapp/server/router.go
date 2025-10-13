package server

import (
	"chemplusx.com/midas/controller"
	"github.com/gin-gonic/gin"
	// "chemplusx.com/midas/neo4j"
)

func InitRouter(r *gin.Engine) {

	r.StaticFile("/home", "./static/templates/home2.html")
	r.StaticFile("/midas", "./static/templates/midas2.html")
	r.StaticFile("/source", "./static/templates/sources2.html")
	// r.StaticFile("/about", "./static/templates/about.html")
	// r.StaticFile("/contact", "./static/templates/contact.html")

	r.Static("/static", "./static/assets")
	r.Static("/js", "./static/js")
	r.Static("/images", "./static/images")
	r.Static("/css", "./static/css")

	r.POST("/api/stream", controller.MIDASHandler)

	// Use the custom OpenAPI request validator middleware in Gin
	// r.Use(middlewares.OapiRequestValidatorWithGin(router))
	InitDataEndpoint(r)
}

func InitDataEndpoint(r *gin.Engine) {
	// client, err := neo4j.NewClient("bolt://localhost:7687", "neo4j", "password")
	// if err != nil {
	// 	log.Fatalf("Failed to create Neo4j client: %v", err)
	// }

	// r.GET("/ws", HandleWebSocket)

	// // Define your API routes
	// r.GET("/list/labels", controllers.GetLabelsHandler(client))
	// r.GET("/list/relations", controllers.GetEdgesHandler(client))
	// r.GET("/list/labels/:label", controllers.GetLabelDetailsHandler(client))
	// r.POST("/nodes", controllers.GetNodesByRequestHandler(client))
	// r.GET("/nodes/search", controllers.SearchNodesHandler(client))
	// r.GET("/nodes/graph", controllers.GetNetworkGraphForIdHandler(client))
	// r.GET("/search_in_graph", controllers.SearchNodesInGraphHandler(client))
	// r.POST("/api/global-search", controllers.GlobalSearchHandler(client))
	// r.POST("/api/interaction-search", controllers.InteractionSearchHandler(client))
	// r.POST("/api/path-search", controllers.PathSearchHandler(client))

}
