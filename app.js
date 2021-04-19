const Koa = require('koa2')
const router = require('./router')
const bodyParser = require('koa-bodyparser')

const app = new Koa()

app.use(bodyParser())
app.use( router.routes(), router.allowedMethods() )

app.listen(5050, ()=>{
    console.log('Serve is running at http://localhost:5050')
})