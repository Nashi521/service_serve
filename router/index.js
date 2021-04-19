const Router = require('koa-router')

const router = new Router()

router.post('/', async (ctx)=>{
    console.log(ctx.request.body)
})


module.exports = router