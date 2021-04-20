const Router = require("koa-router");
const fs = require("fs");
const cp = require("child_process");
const xlsx = require("node-xlsx");

const router = new Router();

router.post("/1", async (ctx) => {
  console.log("Serve get a request 1");
  let form = ctx.request.body;
  let formData = {
    数据大小: form.size,
    数据类型: form.type,
    数据格式: form.format,
    位置地图: form.tag1 ? 1 : 0,
    新闻资讯: form.tag2 ? 1 : 0,
    舆情监测: form.tag3 ? 1 : 0,
    产业经济: form.tag4 ? 1 : 0,
    市内交通: form.tag5 ? 1 : 0,
    智能客服: form.tag6 ? 1 : 0,
    企业工商: form.tag7 ? 1 : 0,
    企业图谱: form.tag8 ? 1 : 0,
    自然灾害: form.tag9 ? 1 : 0,
    智能识别: form.tag10 ? 1 : 0,
    电子商务: form.tag11 ? 1 : 0,
    企业综合: form.tag12 ? 1 : 0,
    环境质量: form.tag13 ? 1 : 0,
    投融资: form.tag14 ? 1 : 0,
    商品信息: form.tag15 ? 1 : 0,
    市场调研: form.tag16 ? 1 : 0,
    公路铁路: form.tag17 ? 1 : 0,
    经营管理: form.tag18 ? 1 : 0,
    公告文书: form.tag19 ? 1 : 0,
    app应用: form.tag20 ? 1 : 0,
    知识产权: form.tag21 ? 1 : 0,
    天气查询: form.tag22 ? 1 : 0,
    信用评估: form.tag23 ? 1 : 0,
    资质备案: form.tag24 ? 1 : 0,
  };
  let excelData = [];
  let title = [];
  let row = [];

  for (let key in formData) {
    title.push(key);
    row.push(formData[key]);
  }
  excelData.push(title, row);

  let buffer = xlsx.build([
    {
      name: "sheet1",
      data: excelData,
    },
  ]);

  //将文件内容插入新的文件中
  fs.writeFileSync("./excel/test1.xlsx", buffer, { flag: "w" });

  let workProcess = cp.exec( "python ./py/image.py", (error, stdout, stderr) => {
      if (error) {
        console.log(error);
      }
      console.log(stdout);
    }
  );
  workProcess.on("exit", (code) => {
    console.log("子进程退出，退出码" + code);
  });

  ctx.body = "1 success";
});

router.post("/2", async (ctx) => {
  console.log("Serve get a request 2");
  let form = ctx.request.body;
  let formData = {
    数据大小: form.size,
    数据类型: form.type,
    数据格式: form.format,
    市场调研: form.tag1 ? 1 : 0,
    产业经济: form.tag2 ? 1 : 0,
    经营管理: form.tag3 ? 1 : 0,
    智能投顾: form.tag4 ? 1 : 0,
    车辆信息: form.tag5 ? 1 : 0,
    商品信息: form.tag6 ? 1 : 0,
    海关进出口: form.tag7 ? 1 : 0,
    知识产权: form.tag8 ? 1 : 0,
    企业综合: form.tag9 ? 1 : 0,
    电子商务: form.tag10 ? 1 : 0,
  };
  let excelData = [];
  let title = [];
  let row = [];

  for (let key in formData) {
    title.push(key);
    row.push(formData[key]);
  }
  excelData.push(title, row);

  let buffer = xlsx.build([
    {
      name: "sheet1",
      data: excelData,
    },
  ]);

  //将文件内容插入新的文件中
  fs.writeFileSync("./excel/test2.xlsx", buffer, { flag: "w" });

  let workProcess = cp.exec( "python ./py/image.py", (error, stdout, stderr) => {
      if (error) {
        console.log(error);
      }
      console.log(stdout);
    }
  );
  workProcess.on("exit", (code) => {
    console.log("子进程退出，退出码" + code);
  });

  ctx.body = "2 success";
});

router.post("/3", async (ctx) => {
  console.log("Serve get a request 3");
  let form = ctx.request.body;
  let formData = {
    智能识别: form.tag1 ? 1 : 0,
    产业经济: form.tag2 ? 1 : 0,
    电子商务: form.tag3 ? 1 : 0,
    商品信息: form.tag4 ? 1 : 0,
    短信API: form.tag5 ? 1 : 0,
    企业综合: form.tag6 ? 1 : 0,
    企业工商: form.tag7 ? 1 : 0,
    银行卡核验: form.tag8 ? 1 : 0,
    车辆信息: form.tag9 ? 1 : 0,
    智能风控: form.tag10 ? 1 : 0,
    天气查询: form.tag11 ? 1 : 0,
    经营管理: form.tag12 ? 1 : 0,
    身份核验: form.tag13 ? 1 : 0,
    知识产权: form.tag14 ? 1 : 0,
    app应用: form.tag15 ? 1 : 0,
    快递查询: form.tag16 ? 1 : 0,
    舆情监测: form.tag17 ? 1 : 0,
    IP地址: form.tag18 ? 1 : 0,
    手机号验证: form.tag19 ? 1 : 0,
    海关进出口: form.tag20 ? 1 : 0,
    智能客服: form.tag21 ? 1 : 0,
    "1分钱": form.tag22 ? 1 : 0,
    司法: form.tag23 ? 1 : 0,
    银行卡信息: form.tag24 ? 1 : 0,
    交通违章: form.tag25 ? 1 : 0,
    资质备案: form.tag26 ? 1 : 0,
    行政监管: form.tag27 ? 1 : 0,
    位置地图: form.tag28 ? 1 : 0,
    招投标: form.tag29 ? 1 : 0,
    公告文书: form.tag30 ? 1 : 0,
    投融资: form.tag31 ? 1 : 0,
    黑名单: form.tag32 ? 1 : 0,
    风控: form.tag33 ? 1 : 0,
    手机号码归属: form.tag34 ? 1 : 0,
    反欺诈: form.tag35 ? 1 : 0,
    星座运势: form.tag36 ? 1 : 0,
    手机号码状态: form.tag37 ? 1 : 0,
    航空航班: form.tag38 ? 1 : 0,
    新闻资讯: form.tag39 ? 1 : 0,
    环境质量: form.tag40 ? 1 : 0,
    尾号限行: form.tag41 ? 1 : 0,
    行驶驾驶: form.tag42 ? 1 : 0,
    税务信息: form.tag43 ? 1 : 0,
    信用评估: form.tag44 ? 1 : 0,
    基站: form.tag45 ? 1 : 0,
    自然灾害: form.tag46 ? 1 : 0,
    万年历: form.tag47 ? 1 : 0,
    手机在网时长: form.tag48 ? 1 : 0,
    企业图谱: form.tag49 ? 1 : 0,
    智能营销: form.tag50 ? 1 : 0,
    油价查询: form.tag51 ? 1 : 0,
    彩票信息: form.tag52 ? 1 : 0,
    公路铁路: form.tag53 ? 1 : 0,
    京东E卡: form.tag54 ? 1 : 0,
    市场调研: form.tag55 ? 1 : 0,
    借贷: form.tag56 ? 1 : 0,
    智能支付: form.tag57 ? 1 : 0,
    区号查询: form.tag58 ? 1 : 0,
    用户画像: form.tag59 ? 1 : 0,
    股票汇率: form.tag60 ? 1 : 0,
    视频会员: form.tag61 ? 1 : 0,
  };
  let excelData = [];
  let title = [];
  let row = [];

  for (let key in formData) {
    title.push(key);
    row.push(formData[key]);
  }
  excelData.push(title, row);

  let buffer = xlsx.build([
    {
      name: "sheet1",
      data: excelData,
    },
  ]);

  //将文件内容插入新的文件中
  fs.writeFileSync("./excel/test3.xlsx", buffer, { flag: "w" });

  let workProcess = cp.exec( "python ./py/image.py", (error, stdout, stderr) => {
      if (error) {
        console.log(error);
      }
      console.log(stdout);
    }
  );
  workProcess.on("exit", (code) => {
    console.log("子进程退出，退出码" + code);
  });

  ctx.body = "3 success";
});

router.post('/uploadfile', async (ctx, next) => {
  // 上传单个文件
  const file = ctx.request.files.file; // 获取上传文件
  // 创建可读流
  const reader = fs.createReadStream(file.path);
  let filePath = `./excel/test1.xlsx`;
  // 创建可写流
  const upStream = fs.createWriteStream(filePath);
  // 可读流通过管道写入可写流
  reader.pipe(upStream);
  return ctx.body = "上传成功！";
});

module.exports = router;
