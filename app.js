const Koa = require('koa2')
const router = require('./router')
const cors = require('koa2-cors')
const koaBody = require('koa-body');

const app = new Koa()

app.use(cors());
app.use(koaBody({
    multipart: true,
    formidable: {
        maxFileSize: 200 * 1024 * 1024    // 设置上传文件大小最大限制，默认2M
    }
}));
app.use(router.routes(), router.allowedMethods())

app.listen(5050, () => {
    console.log('Serve is running at http://localhost:5050')
})