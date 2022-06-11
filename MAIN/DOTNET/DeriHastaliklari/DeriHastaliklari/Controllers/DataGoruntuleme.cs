using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace DeriHastaliklari.Controllers
{
    public class DataGoruntuleme : Controller
    {
        [Authorize]
        public IActionResult Index()
        {
            return View();
        }

       
        
    }
}
