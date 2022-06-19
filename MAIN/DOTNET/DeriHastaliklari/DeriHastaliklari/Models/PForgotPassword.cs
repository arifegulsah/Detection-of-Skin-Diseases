using System.ComponentModel.DataAnnotations;

namespace DeriHastaliklari.Models
{
    public class PForgotPassword
    {

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Name { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Surname { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string TcNo { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string bthDay { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Email { get; set; }



    }
}
